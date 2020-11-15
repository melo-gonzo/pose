from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import smtplib
import imaplib
import email
import time
import os
from inference import *


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

    def send_email(self, angle_text, attachment_path=''):
        attachment = open(attachment_path, 'rb').read()
        documentation = open('/home/carmelo/Documents/pose/helpful_documents/angleeye_quickstart.pdf', 'rb').read()
        msg = MIMEMultipart()
        msg['Subject'] = 'Angle Eye AI Results!'
        msg['To'] = self.output_email
        msg['From'] = self.username
        body_message = open('/home/carmelo/Documents/pose/helpful_documents/body_message.txt').read()
        body_text = 'Hey there ' + self.output_name.split(' ')[0] + body_message
        body_text = body_text + angle_text if angle_text is not None else body_text
        text = MIMEText(body_text)
        msg.attach(text)
        pdf = MIMEApplication(documentation)
        fname = 'Quick Start Guide.pdf'
        pdf.add_header('Content-Disposition', 'attachment', name=fname, filename=fname)
        msg.attach(pdf)
        image = MIMEImage(attachment, name=os.path.basename(attachment_path))
        msg.attach(image)
        time.sleep(0.1)
        with smtplib.SMTP(self.mail_server, port=587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(msg["From"], self.password)
            time.sleep(0.1)
            smtp.send_message(msg)
            smtp.quit()
        print('Message Sent Successfully!')

    def gen_angle_text(self, angles):
        text_top = '\n\nWe were able to figure out how your limbs are positioned relative to one another.' \
                   ' Here are some of the key measurements we came up with.\n\n'
        keys = angles.keys()
        text = "{:<20} {:<15} {:<10}".format('Pair', 'Joint-Numbers', 'Angle') + '\n'
        for key in keys:
            key_text = key.split(' ')
            key_value = str(np.abs(angles[key])[0])
            f = "{:<20} {:<15} {:<10}".format(key_text[0], key_text[1], key_value)
            text = text + f + '\n'
        text_top = text_top + '\nEnjoy your data!'
        # v = open('results.txt', 'w')
        # v.write(text)
        # v.close()
        return text_top, text

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
            download_folder = download_folder + self.output_name
            if not os.path.isdir(download_folder):
                os.mkdir(download_folder)
            filename = part.get_filename()
            filename = self.output_name.strip(' ') + filename
            fname = filename.split('.')
            filename = ''.join(fname[:-1]) + '.' + fname[-1]
            att_path = os.path.join(download_folder, filename)
            if os.path.isfile(att_path):
                filename = ''.join(fname[:-1]) + time.strftime("%Y-%m-%dT%H%M%S", time.localtime()) + '.' + fname[-1]
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
                    print("No new emails to read.", end='\r')
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
        print(info_tuple)
        return info_tuple


def do_email_thang():
    from credentials import username, password
    while True:
        try:
            AngleEye = FetchEmail(mail_server="imap.gmail.com",
                                   username=username,
                                   password=password)
            emails = AngleEye.fetch_unread_messages()
            if emails is not None:
                for email in emails:
                    print('Getting name and email address')
                    AngleEye.set_name_and_email(AngleEye.parse_email_address(email))
                    print('Saving attachment')
                    att_path = AngleEye.save_attachment(email)
                    if att_path is not None:
                        print('Performing inference')
                        inference_path = inference(att_path)
                        print('Sending email to user')
                        top_text = None
                        AngleEye.send_email(top_text, inference_path)
                        print('Sent to: ' + str(AngleEye.parse_email_address(email)) + ' at time: '
                              + time.strftime("%Y-%m-%dT%H%M%S", time.localtime()))
        except Exception:
            print('Problem! ' + time.strftime("%Y-%m-%dT%H%M%S", time.localtime()))
            print(traceback.format_exc())
            pass
        print('Sleeping. ' + time.strftime("%Y-%m-%dT%H%M%S", time.localtime()), end='\r')
        time.sleep(59)
        try:
            AngleEye.close_connection()
        except Exception:
            print('Problem! ' + time.strftime("%Y-%m-%dT%H%M%S", time.localtime()))
            print(traceback.format_exc())
            pass
        time.sleep(1)
    # frame, points = inference(media_path)


do_email_thang()
