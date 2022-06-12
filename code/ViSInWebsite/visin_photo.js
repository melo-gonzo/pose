import {
  createAngleTable,
  updateAngleTable,
  getAngleTable,
} from "./visin_stats.js";
import { Upload } from "./camera.js";
import { setupDatGui } from "./option_panel.js";
import { STATE } from "./params.js";
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
  // await checkGuiUpdate();
  // detector = await createDetector();
}

async function loadImageAndInference() {
  await checkGuiUpdate();
  detector = await createDetector();
  const image = await loadImage();
  const input = tf.browser.fromPixels(image);
  console.log(detector)
  const poses = await detector.estimatePoses(image, {
    maxPoses: STATE.modelConfig.maxPoses,
    flipHorizontal: false,
  });
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
    angle_dict = getAngleTable(poses);
    updateAngleTable(angle_dict);
  }
  // camera.drawCtx(image);
  input.dispose();
}

async function loadImage() {
  const image = new Image();
  var resize_ = 512;
  const promise = new Promise((resolve, reject) => {
    var item = document.getElementById("file").files[0];
    var reader = new FileReader();
    reader.readAsDataURL(item);
    reader.name = item.name; //get the image's name
    reader.size = item.size; //get the image's size
    reader.onload = function (event) {
      image.src = event.target.result;
      image.size = event.target.size; //set size (optional)
      image.crossOrigin = "";
      image.onload = function (e) {
        var elem = document.getElementById("output");
        if (e.target.height > e.target.width) {
          var scaleFactor = resize_ / e.target.height;
          elem.height = resize_;
          elem.width = e.target.width * scaleFactor;
        } else {
          var scaleFactor = resize_ / e.target.width;
          elem.width = resize_;
          elem.height = e.target.height * scaleFactor;
        }
        var ctx = elem.getContext("2d");
        ctx.drawImage(e.target, 0, 0, elem.width, elem.height);
        // camera.drawCtx(e.target)
        image.width = elem.width;
        image.height = elem.height;
        resolve(image);
      };
    };
  });
  return promise;
}

let btn = document.querySelector("button");
btn.addEventListener("click", function () {
  loadImageAndInference();
});


createAngleTable();
app();