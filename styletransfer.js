"use strict";

console.log("StyleTransfer: Using Tensorflow.js version " + tf.version.tfjs);

var canvas = document.getElementById("canvas");
canvas.style.display = "none";

var img = new Image();
img.src = "stata.jpg";
// img.style.display = "inline-block";
// document.body.insertBefore(img, canvas);

var data;

var canvas1 = document.createElement("canvas");
var context = canvas1.getContext("2d");

img.onload = function() {
  canvas1.width = img.width;
  canvas1.height = img.height;
  context.drawImage(img, 0, 0);
  data = context.getImageData(0, 0, img.width, img.height);
}

// document.body.appendChild(canvas1);

var styleTransferWorker = new Worker("worker.js");

function style(style) {
  //tf.setBackend("webgl");
  console.log("style " + tf.getBackend());
  // styleTransferWorker.postMessage([style, JSON.stringify(Array.from(await tf.fromPixels(img).data())), tf.fromPixels(img).shape]);
  // styleTransferWorker.postMessage([style]);
  styleTransferWorker.postMessage([style, data]);
}

function setCanvasShape() {
  canvas.width = 150;//shape[1];
  canvas.height = 100;//shape[0];
  // if (shape[1] > shape[0]) {
  //   canvas.style.width = "500px";
  //   canvas.style.height = (shape[0] / shape[1] * 500) + "px";
  // } else {
  //   canvas.style.height = "500px";
  //   canvas.style.width = (shape[1] / shape[0] * 500) + "px";
  // }
}

function renderToCanvas(imageData, canvas) {
  canvas.style.display = "inline-block";
  const ctx = canvas.getContext("2d");
  ctx.putImageData(imageData, 0, 0);
}

styleTransferWorker.onmessage = function(e) {
  console.log("styletransfer onmessage " + tf.getBackend());
  setCanvasShape();
  renderToCanvas(e.data, canvas);
}
