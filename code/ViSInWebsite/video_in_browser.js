const frames = [];
const button = document.querySelector("button");
const select = document.querySelector("select");
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");

button.onclick = async(evt) => {
  if (HTMLVideoElement.prototype.requestVideoFrameCallback) {
    let stopped = false;
    const video = await getVideoElement();
    const drawingLoop = async(timestamp, frame) => {
      const bitmap = await createImageBitmap(video);
      const index = frames.length;
      frames.push(bitmap);
      select.append(new Option("Frame #" + (index + 1), index));

      if (!video.ended && !stopped) {
        video.requestVideoFrameCallback(drawingLoop);
      } else {
        select.disabled = false;
      }
    };
    video.requestVideoFrameCallback(drawingLoop);
    button.onclick = (evt) => stopped = true;
    button.textContent = "stop";
  } else {
    console.error("your browser doesn't support this API yet");
  }
};

select.onchange = (evt) => {
  const frame = frames[select.value];
  canvas.width = frame.width;
  canvas.height = frame.height;
  ctx.drawImage(frame, 0, 0);
};

async function getVideoElement() {
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.src = "https://upload.wikimedia.org/wikipedia/commons/a/a4/BBH_gravitational_lensing_of_gw150914.webm";
  document.body.append(video);
  await video.play();
  return video;
}
