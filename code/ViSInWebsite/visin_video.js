import {
  createAngleTable,
  updateAngleTable,
  getAngleTable,
} from "./visin_stats.js";
import { Upload } from "./camera.js";
import { setupDatGui } from "./option_panel.js";
import { STATE, VIDEO_SIZE } from "./params.js";
import { setBackendAndEnvFlags } from "./util.js";
let detector, camera, stats;
let startInferenceTime,
  numInferences = 0;
let inferenceTimeSum = 0,
  lastPanelUpdate = 0;
let rafId;
let angle_dict;

async function createDetector() {
  switch (STATE.model) {
    case poseDetection.SupportedModels.PoseNet:
      return poseDetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: "MobileNetV1",
        outputStride: 16,
        inputResolution: { width: 500, height: 500 },
        multiplier: 0.75,
      });
    case poseDetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split("-")[0];
      if (runtime === "mediapipe") {
        return poseDetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/pose",
        });
      } else if (runtime === "tfjs") {
        return poseDetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
        });
      }
    case poseDetection.SupportedModels.MoveNet:
      const modelType =
        STATE.modelConfig.type == "lightning"
          ? poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
          : poseDetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return poseDetection.createDetector(STATE.model, { modelType });
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Upload.setupPhoto(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;
    window.cancelAnimationFrame(rafId);
    detector.dispose();
    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }
    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

async function app() {
  const urlParams = new URLSearchParams(window.location.search);
  await setupDatGui(urlParams);
  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  camera = await Upload.setupPhoto(STATE.camera);
  await checkGuiUpdate();
  detector = await createDetector();
}

async function inferenceFromVideoFrame(input_frame) {
  
  const input = tf.browser.fromPixels(input_frame);
  const poses = await detector.estimatePoses(input, {
    maxPoses: STATE.modelConfig.maxPoses,
    flipHorizontal: false,
  });
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
    angle_dict = getAngleTable(poses);
    updateAngleTable(angle_dict);
  }
  input.dispose();
  return {poses, angle_dict}
}

async function resizeImage(input_image) {
  const image = new Image();
  var resize_ = 512;
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = "";
    var elem = document.getElementById("output");
    if (input_image.height > input_image.width) {
      var scaleFactor = resize_ / input_image.height;
      elem.height = resize_;
      elem.width = input_image.width * scaleFactor;
    } else {
      var scaleFactor = resize_ / input_image.width;
      elem.width = resize_;
      elem.height = input_image.height * scaleFactor;
    }
    var ctx = elem.getContext("2d");
    ctx.drawImage(input_image, 0, 0, elem.width, elem.height);
    image.width = elem.width;
    image.height = elem.height;
    image.src = elem.toDataURL(elem.type);
    resolve(image);
  });
  return promise;
}

createAngleTable();
app();

const frames = [];
const poses = [];
const angle_dicts = [];
const button = document.querySelector("button");
const select = document.querySelector("select");
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");

button.onclick = async (evt) => {
  if (HTMLVideoElement.prototype.requestVideoFrameCallback) {
    let stopped = false;
    const video = await getVideoElement();
    const drawingLoop = async (timestamp, frame) => {
      const index = frames.length;
      select.append(new Option("Frame #" + (index + 1), index));
      const input_image = await resizeImage(video);
      const output = await inferenceFromVideoFrame(input_image);
      frames.push(input_image);
      poses.push(output.poses)
      angle_dicts.push(output.angle_dict)
      if (!video.ended && !stopped) {
        video.requestVideoFrameCallback(drawingLoop);
      } else {
        select.disabled = false;
      }
    };
    video.requestVideoFrameCallback(drawingLoop);
    button.onclick = (evt) => (stopped = true);
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
  if (poses[select.value].length > 0) {
    camera.drawResults(poses[select.value]);
    const angle_dict11 = getAngleTable(poses[select.value]);
    updateAngleTable(angle_dicts[select.value]);
  }
};

async function getVideoElement() {
  const video = document.getElementById("video");
  video.crossOrigin = "anonymous";
  var item = document.getElementById("file").files[0];
  var reader = new FileReader();
  reader.readAsDataURL(item);
  reader.name = item.name;
  reader.size = item.size;
  reader.onload = function (event) {
    video.src = event.target.result;
  };
  document.body.append(video);
  await video.play();
  return video;
}
