from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.message import EmailMessage
from email.header import decode_header
from email.mime.text import MIMEText
import webbrowser
import smtplib
import imaplib
import email
import time
import ssl
import os
from keypoint_analysis import *


class FetchEmail():
    connection = None
    error = None

    def __init__(self, mail_server, username, password):
        self.connection = imaplib.IMAP4_SSL(mail_server)
        self.connection.login(username, password)
        self.connection.select(readonly=False)  # so we can mark mails as read
        self.username = username
        self.password = password
        self.mail_server = mail_server

    def set_name_and_email(self, info_tuple):
        self.output_name = info_tuple[0]
        self.output_email = info_tuple[1]

    def close_connection(self):
        """
        Close the connection to the IMAP server
        """
        self.connection.close()

    def send_email(self, attachment_path=''):
        attachment = open(attachment_path, 'rb').read()
        msg = MIMEMultipart()
        msg['Subject'] = 'AI Bike Fit Results!'
        msg['To'] = self.output_email
        msg['From'] = 'ai.bikefit@gmail.com'
        text = MIMEText('Hey there ' + self.output_name + '. You\'re one of the first users of AI Bike Fit! I\'m trying to draw a stick figure on you. Hopefully it turned out ok. :)')
        # text = MIMEText('Yeehaw!')
        msg.attach(text)
        image = MIMEImage(attachment, name=os.path.basename(attachment_path))
        msg.attach(image)
        # context = ssl.create_default_context()
        time.sleep(0.1)
        with smtplib.SMTP(self.mail_server, port=587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(msg["From"], self.password)
            smtp.send_message(msg)
            smtp.quit()

    def save_attachment(self, msg,
                        download_folder='/home/carmelo/Documents/pose/data_processing/ai_bikefit_downloads/'):
        """
        Given a message, save its attachments to the specified
        download folder (default is /tmp)
        return: file path to attachment
        """
        att_path = None
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue

            filename = part.get_filename()
            filename = self.output_name.strip(' ') + filename
            att_path = os.path.join(download_folder, filename)

            if not os.path.isfile(att_path):
                fp = open(att_path, 'wb')
                fp.write(part.get_payload(decode=True))
                fp.close()
        return att_path

    def fetch_unread_messages(self):
        """
        Retrieve unread messages
        """
        emails = []
        (result, messages) = self.connection.search(None, 'UnSeen')
        messages = messages[0].decode().split(' ')
        if result == "OK":
            for message in messages:
                try:
                    ret, data = self.connection.fetch(message, '(RFC822)')
                except:
                    print("No new emails to read.")
                    return None
                    # self.close_connection()
                    # exit()
                msg = email.message_from_bytes(data[0][1])
                if isinstance(msg, str) == False:
                    emails.append(msg)
                response, data = self.connection.store(message, '+FLAGS', '\\Seen')

            return emails

        self.error = "Failed to retrieve emails."
        return emails

    def parse_email_address(self, email_address):
        """
        Helper function to parse out the email address from the message

        return: tuple (name, address). Eg. ('John Doe', 'jdoe@example.com')
        """
        email_address = email_address.get("From")
        ename = email_address.find('<')
        info_tuple = (email_address[:ename], email_address[ename + 1:-1])
        return info_tuple


def do_email_thang():
    username = "self-explanatory@gmail.com"
    password = "sikeYOUthought"
    while True:
        AIBikeFit = FetchEmail(mail_server="imap.gmail.com",
                               username=username,
                               password=password)
        emails = AIBikeFit.fetch_unread_messages()
        print(emails)
        if emails is not None:
            for email in emails:
                print('Getting name and email address')
                AIBikeFit.set_name_and_email(AIBikeFit.parse_email_address(email))
                print('Saving attachment')
                att_path = AIBikeFit.save_attachment(email)
                if att_path is not None:
                    print('Performing inference')
                    frame, points = inference(att_path)
                    inference_path = att_path.split('.')
                    inference_path = inference_path[0] + '_inference.' + inference_path[1]
                    print('Saving inference image')
                    cv2.imwrite(inference_path, frame)
                    print('Sending email to user')
                    AIBikeFit.send_email(inference_path)
                    print('Sent to: ' + str(AIBikeFit.parse_email_address(email)))
        time.sleep(29)
        AIBikeFit.close_connection()
        time.sleep(1)
    # frame, points = inference(media_path)


do_email_thang()
