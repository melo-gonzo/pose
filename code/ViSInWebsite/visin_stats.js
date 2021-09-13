const body_parts = {
  Nose: 0,
  leftEye: 1,
  rightEye: 2,
  leftEar: 3,
  rightEar: 4,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
};

const pairs = [
  ["rightShoulder", "rightElbow"],
  ["rightElbow", "rightWrist"],
  ["leftShoulder", "leftElbow"],
  ["leftElbow", "leftWrist"],
  ["rightHip", "rightKnee"],
  ["rightKnee", "rightAnkle"],
  ["leftHip", "leftKnee"],
  ["leftKnee", "leftAnkle"],
  ["Nose", "rightEye"],
  ["rightEye", "rightEar"],
  ["Nose", "leftEye"],
  ["LEye", "leftEar"],
];

// const angle_pairs = [
//     [["rightShoulder", "rightElbow"],["rightElbow", "rightWrist"]],
//     [["leftShoulder", "leftElbow"],["leftElbow", "leftWrist"]],
//     [["rightHip", "rightKnee"],["rightKnee", "rightAnkle"]],
//     [["leftHip", "leftKnee"],["leftKnee", "leftAnkle"]],
//     [["rightShoulder", "rightElbow"], ["Vert"]],
//     [["rightElbow", "rightWrist"], ["Vert"]],
//     [["leftShoulder", "leftElbow"], ["Vert"]],
//     [["leftElbow", "leftWrist"], ["Vert"]],
//     [["rightHip", "rightKnee"], ["Vert"]],
//     [["rightKnee", "rightAnkle"], ["Vert"]],
//     [["leftHip", "leftKnee"], ["Vert"]],
//     [["leftKnee", "leftAnkle"], ["Vert"]],
//     [["leftHip", "rightHip"], ["Horiz"]],
//     [["leftShoulder", "rightShoulder"], ["Horiz"]],
//   ];

const angle_pairs = [
  [
    ["rightShoulder", "rightElbow"],
    ["rightElbow", "rightWrist"],
  ],
  [
    ["leftShoulder", "leftElbow"],
    ["leftElbow", "leftWrist"],
  ],
  [
    ["rightHip", "rightKnee"],
    ["rightKnee", "rightAnkle"],
  ],
  [
    ["leftHip", "leftKnee"],
    ["leftKnee", "leftAnkle"],
  ],
  [["rightShoulder", "rightElbow"], ["Vert"]],
  [["rightElbow", "rightWrist"], ["Vert"]],
  [["leftShoulder", "leftElbow"], ["Vert"]],
  [["leftElbow", "leftWrist"], ["Vert"]],
  [["rightHip", "rightKnee"], ["Vert"]],
  [["rightKnee", "rightAnkle"], ["Vert"]],
  [["leftHip", "leftKnee"], ["Vert"]],
  [["leftKnee", "leftAnkle"], ["Vert"]],
  [["leftHip", "rightHip"], ["Horiz"]],
  [["leftShoulder", "rightShoulder"], ["Horiz"]],
];

export function getAngleTable(predictedPoses) {
  const angle_dict = {};
  for (var i = 0; i < angle_pairs.length; i++) {
    let x1, x2, x3, x4, y1, y2, y3, y4, v1x, v1y, v2x, v2y;
    if (angle_pairs[i][0] == "Vert") {
      x1 = 0;
      x2 = 0;
      y1 = 0;
      y2 = 1;
      v1x = 0;
      v1y = 1;
      x3 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][0]]].x;
      x4 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][1]]].x;
      v2x = x4 - x3;
      y3 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][0]]].y;
      y4 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][1]]].y;
      v2y = y4 - y3;
    } else if (angle_pairs[i][1] == "Vert") {
      x3 = 0;
      x4 = 0;
      y3 = 0;
      y4 = 1;
      v2x = 0;
      v2y = 1;
      x1 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][0]]].x;
      x2 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][1]]].x;
      v1x = x2 - x1;
      y1 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][0]]].y;
      y2 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][1]]].y;
      v1y = y2 - y1;
    } else if (angle_pairs[i][0] == "Horiz") {
      x1 = 0;
      x2 = 1;
      y1 = 0;
      y2 = 0;
      v1x = 1;
      v1y = 0;
      x3 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][0]]].x;
      x4 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][1]]].x;
      v2x = x4 - x3;
      y3 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][0]]].y;
      y4 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][1]]].y;
      v2y = y4 - y3;
    } else if (angle_pairs[i][1] == "Horiz") {
      x3 = 0;
      x4 = 1;
      y3 = 0;
      y4 = 0;
      v2x = 1;
      v2y = 0;
      x1 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][0]]].x;
      x2 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][1]]].x;
      v1x = x2 - x1;
      y1 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][0]]].y;
      y2 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][1]]].y;
      v1y = y2 - y1;
    } else {
      x1 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][0]]].x;
      x2 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][1]]].x;
      v1x = x2 - x1;
      y1 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][0]]].y;
      y2 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][0][1]]].y;
      v1y = y2 - y1;
      x3 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][0]]].x;
      x4 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][1]]].x;
      v2x = x4 - x3;
      y3 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][0]]].y;
      y4 = predictedPoses[0].keypoints[body_parts[angle_pairs[i][1][1]]].y;
      v2y = y4 - y3;
    }
    let angle = Math.abs(
      Math.round(
        100 * ((180 / Math.PI) * (Math.atan2(v1y, v1x) - Math.atan2(v2y, v2x)))
      ) / 100
    );
    let opposite_button = "opposite_button" + String(i);
    let oppositeButton = document.getElementById(opposite_button).value;
    if (oppositeButton == "Reverse") {
      angle = Math.abs(Math.round(100 * (360 - angle)) / 100);
    }
    const angel_between = [
      angle_pairs[i][0].join("-"),
      angle_pairs[i][1].join("-"),
    ].join(":");
    angle_dict[angel_between] = angle;
    const showButton = "show_button" + String(i);
    let buttonOn = document.getElementById(showButton).value;
    if (buttonOn == "On") {
      var canvas = document.getElementById("output");
      var context = canvas.getContext("2d");
      var text = angle;
      context.font = "25px Arial";
      context.fillStyle = "0,0,0";
      let x, y;
      if (
        x1 != 0 &&
        x2 != 0 &&
        x3 != 0 &&
        x4 != 0 &&
        y1 != 0 &&
        y2 != 0 &&
        y3 != 0 &&
        y4 != 0
      ) {
        x = find_mode([x1, x2, x3, x4]);
        y = find_mode([y1, y2, y3, y4]);
      }
      if (x1 == 0 || x2 == 0) {
        x =
          (Math.max(...[x3, x4]) - Math.min(...[x3, x4])) / 2 +
          Math.min(...[x3, x4]);
        y =
          (Math.max(...[y3, y4]) - Math.min(...[y3, y4])) / 2 +
          Math.min(...[y3, y4]);
      } else if (x3 == 0 || x4 == 0) {
        x =
          (Math.max(...[x1, x2]) - Math.min(...[x1, x2])) / 2 +
          Math.min(...[x1, x2]);
        y =
          (Math.max(...[y1, y2]) - Math.min(...[y1, y2])) / 2 +
          Math.min(...[y1, y2]);
      }
      if (y1 == 0 || y2 == 0) {
        y =
          (Math.max(...[y3, y4]) - Math.min(...[y3, y4])) / 2 +
          Math.min(...[y3, y4]);
        x =
          (Math.max(...[x3, x4]) - Math.min(...[x3, x4])) / 2 +
          Math.min(...[x3, x4]);
      } else if (y3 == 0 || y4 == 0) {
        y =
          (Math.max(...[y1, y2]) - Math.min(...[y1, y2])) / 2 +
          Math.min(...[y1, y2]);
        x =
          (Math.max(...[x1, x2]) - Math.min(...[x1, x2])) / 2 +
          Math.min(...[x1, x2]);
      }
      context.fillText(text, x, y);
      const circle = new Path2D();
      circle.arc(x, y, 4, 0, 2 * Math.PI);
      context.fillStyle = "Black";
      context.fill(circle);
      context.stroke(circle);
    }
  }
  return [angle_dict];
}

export function createAngleTable() {
  let table = document.querySelector("table");
  let thead = table.createTHead();
  let row = thead.insertRow();
  let data = ["Show", "Opposite", "Pair", "Angle"];
  for (let key of data) {
    let th = document.createElement("th");
    let text = document.createTextNode(key);
    th.appendChild(text);
    row.appendChild(th);
  }
  const angle_dict = {};
  for (var i = 0; i < angle_pairs.length; i++) {
    const angel_between = [
      angle_pairs[i][0].join("-"),
      angle_pairs[i][1].join("-"),
    ].join(":");
    angle_dict[angel_between] = 0;
  }
  const angle_keys = Object.keys(angle_dict);
  var show_buttons = [];
  var opposite_buttons = [];
  for (var i = 0; i < angle_keys.length; i++) {
    let row = table.insertRow();
    let cell0 = row.insertCell();
    var btn0 = document.createElement("input");
    show_buttons.push(btn0);
    show_buttons[i].type = "button";
    show_buttons[i].className = "btn";
    show_buttons[i].value = "Off";
    show_buttons[i].id = "show_button" + String(i);
    show_buttons[i].onclick = (function (btn) {
      return function () {
        switch (btn.value) {
          case "Off":
            btn.value = "On";
            document.getElementById(btn.id).style.backgroundColor = "Green";
            break;
          case "On":
            btn.value = "Off";
            document.getElementById(btn.id).style.backgroundColor = "Red";
            break;
          default:
            btn.value = "Off";
            document.getElementById(btn.id).style.backgroundColor = "Red";
        }
      };
    })(show_buttons[i]);
    cell0.appendChild(show_buttons[i]);
    document.getElementById(show_buttons[i].id).style.backgroundColor = "red";

    let cell1 = row.insertCell();
    var btn1 = document.createElement("input");
    opposite_buttons.push(btn1);
    opposite_buttons[i].type = "button";
    opposite_buttons[i].className = "btn";
    opposite_buttons[i].value = "Regular";
    opposite_buttons[i].id = "opposite_button" + String(i);
    opposite_buttons[i].onclick = (function (btn) {
      return function () {
        switch (btn.value) {
          case "Regular":
            btn.value = "Reverse";
            document.getElementById(btn.id).style.backgroundColor = "Red";
            break;
          case "Reverse":
            btn.value = "Regular";
            document.getElementById(btn.id).style.backgroundColor = "Green";
            break;
          default:
            btn.value = "Regular";
            document.getElementById(btn.id).style.backgroundColor = "Green";
        }
      };
    })(opposite_buttons[i]);
    cell1.appendChild(opposite_buttons[i]);
    document.getElementById(opposite_buttons[i].id).style.backgroundColor =
      "Green";

    let cell2 = row.insertCell();
    let text2 = document.createTextNode([angle_keys[i]]);
    cell2.appendChild(text2);
    let cell3 = row.insertCell();
    let text3 = document.createTextNode([angle_dict[angle_keys[i]]]);
    cell3.appendChild(text3);
  }
}

export function updateAngleTable(angle_dict) {
  const angle_keys = Object.keys(angle_dict[0]);
  let table = document.querySelector("table");
  for (var i = 0; i < angle_keys.length; i++) {
    const new_angle = angle_dict[0][angle_keys[i]];
    table.rows[i + 1].cells[3].innerHTML = new_angle;
  }
}

function find_mode(arr) {
  var mode = {};
  var max = 0,
    count = 0;
  arr.forEach(function (e) {
    if (mode[e]) {
      mode[e]++;
    } else {
      mode[e] = 1;
    }
    if (count < mode[e]) {
      max = e;
      count = mode[e];
    }
  });
  return max;
}
