let playing = false;
let frames = [];
let video;

const select = document.querySelector("select");

function togglePlayback() {
    if (playing) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    resetWorkflow();
    playing = true;
    document.getElementById("imageButton").textContent = "Stop";

    getVideoElement().then((v) => {
        video = v;
        video.addEventListener('canplay', () => {
            const drawingLoop = async (timestamp, frame) => {
                const index = frames.length;
                select.append(new Option("Frame #" + (index + 1), index));
                const input_image = await resizeImage(video);
                frames.push(input_image);
                if (!video.ended && playing) {
                    video.requestVideoFrameCallback(drawingLoop);
                } else {
                    select.disabled = false;
                }
            };
            video.requestVideoFrameCallback(drawingLoop);
        });
    });
}

function stopPlayback() {
    playing = false;
    document.getElementById("imageButton").textContent = "Play";
    if (video) {
        video.pause();
        video = null;
    }
}

function resetWorkflow() {
    frames = [];
    select.innerHTML = '';
    document.getElementById("output").getContext("2d").clearRect(0, 0, 512, 512);
}

async function getVideoElement() {
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    const item = document.getElementById("file").files[0];
    const reader = new FileReader();
    reader.readAsArrayBuffer(item);
    reader.name = item.name;
    reader.size = item.size;
    reader.onload = function (event) {
        const blob = new Blob([event.target.result]);
        video.src = URL.createObjectURL(blob);
        document.body.append(video);
        video.hidden = true;
        video.play();
    };
    return video;
}

select.onchange = (evt) => {
    const selectedIndex = select.value;

    if (selectedIndex >= 0 && selectedIndex < frames.length) {
        const selectedFrame = frames[selectedIndex];
        const videoCanvas = document.getElementById("output");
        const ctx = videoCanvas.getContext("2d");

        // Clear the canvas before drawing the new frame
        ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);

        // Draw the selected frame onto the video canvas
        ctx.drawImage(selectedFrame, 0, 0, videoCanvas.width, videoCanvas.height);
    }
};

async function resizeImage(input_video) {
    const videoCanvas = document.getElementById("output");
    const ctx = videoCanvas.getContext("2d");

    const resize_ = 512;
    const maxWidth = 512;
    const maxHeight = 512;

    return new Promise((resolve, reject) => {
        const widthScaleFactor = maxWidth / input_video.videoWidth;
        const heightScaleFactor = maxHeight / input_video.videoHeight;
        const scaleFactor = Math.min(widthScaleFactor, heightScaleFactor);
        const scaledWidth = input_video.videoWidth * scaleFactor;
        const scaledHeight = input_video.videoHeight * scaleFactor;

        videoCanvas.width = scaledWidth;
        videoCanvas.height = scaledHeight;

        ctx.drawImage(input_video, 0, 0, scaledWidth, scaledHeight);

        const image = new Image();
        image.width = scaledWidth;
        image.height = scaledHeight;
        image.src = videoCanvas.toDataURL("image/png");
        resolve(image);
    });
}
function changeSelectedFrame(offset) {
    const selectedIndex = parseInt(select.value);
    const newIndex = (selectedIndex + offset + frames.length) % frames.length;

    select.value = newIndex;
    select.onchange(); // Trigger the onchange event to update the displayed frame
}