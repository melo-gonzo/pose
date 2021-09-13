import {
  createAngleTable,
  updateAngleTable,
  getAngleTable,
} from "./visin_stats.js";
import { Camera } from "./camera.js";
import { setupDatGui } from "./option_panel.js";
import { STATE } from "./params.js";
import { setupStats } from "./stats_panel.js";
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
        STATE.modelConfig.type == "thunder"
          ? poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
          : poseDetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return poseDetection.createDetector(STATE.model, { modelType });
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged || STATE.isFacingModeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
    STATE.isFacingModeOptionChanged = false;
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

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
      1000.0 / averageInferenceTime,
      120 /* maxValue */
    );
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }
  // FPS only counts the time it takes to finish estimatePoses.
  beginEstimatePosesStats();
  const poses = await detector.estimatePoses(camera.video, {
    maxPoses: STATE.modelConfig.maxPoses,
    flipHorizontal: false,
  });
  endEstimatePosesStats();
  camera.drawCtx();
  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
    angle_dict = getAngleTable(poses);
    updateAngleTable(angle_dict);
  }
}

async function renderPrediction() {
  await checkGuiUpdate();
  if (!STATE.isModelChanged) {
    await renderResult();
  }
  rafId = requestAnimationFrame(renderPrediction);
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  await setupDatGui(urlParams);
  stats = setupStats();
  console.log(stats);
  camera = await Camera.setupCamera(STATE.camera);
  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  detector = await createDetector();
  renderPrediction();
}

createAngleTable();
app();
