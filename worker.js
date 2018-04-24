importScripts("tfjs-0.9.1.js");

//tf.setBackend("cpu");

const MANIFEST_FILE = "manifest.json";
const STYLES = ["la_muse", "rain_princess", "scream", "udnie", "wave", "wreck"];

// TODO: don't use this deprecated nonsense
class CheckpointLoader {
  constructor(urlPath) {
    this.urlPath = urlPath;
    this.variables = null;
    this.checkpointManifest = null;
  }

  loadManifest() {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("GET", this.urlPath + MANIFEST_FILE);
      xhr.onload = () => {
        this.checkpointManifest = JSON.parse(xhr.responseText);
        resolve();
      };
      xhr.onerror = (error) => {
        throw new Error(`${MANIFEST_FILE} not found at ${this.urlPath}. ${error}`);
      };
      xhr.send();
    });
  }

  getCheckpointManifest() {
    if (this.checkpointManifest === null) {
      return new Promise((resolve, reject) => {
        this.loadManifest().then(() => {
          resolve(this.checkpointManifest);
        });
      });
    }
    return new Promise((resolve, reject) => {
      resolve(this.checkpointManifest);
    });
  }

  getAllVariables() {
    if (this.variables !== null) {
      return new Promise((resolve, reject) => {
        resolve(this.variables);
      });
    }
    return new Promise((resolve, reject) => {
      this.getCheckpointManifest().then(checkpointDefinition => {
        const variableNames = Object.keys(this.checkpointManifest);
        const variablePromises = [];
        for (let i = 0; i < variableNames.length; i++) {
          variablePromises.push(this.getVariable(variableNames[i]));
        }
        Promise.all(variablePromises).then(variables => {
          this.variables = {};
          for (let i = 0; i < variables.length; i++) {
            this.variables[variableNames[i]] = variables[i];
          }
          resolve(this.variables);
        });
      });
    });
  }

  getVariable(varName) {
    if (!(varName in this.checkpointManifest)) {
      throw new Error("Cannot load non-existent variable " + varName);
    }
    const variableRequestPromiseMethod = (resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.responseType = "arraybuffer";
      const fname = this.checkpointManifest[varName].filename;
      xhr.open("GET", this.urlPath + fname);
      xhr.onload = () => {
        if (xhr.status === 404) {
          throw new Error(`Not found variable ${varName}`);
        }
        const values = new Float32Array(xhr.response);
        const tensor = tf.tensor(Array.from(values), this.checkpointManifest[varName].shape);
        resolve(tensor);
      };
      xhr.onerror = (error) => {
        throw new Error(`Could not fetch variable ${varName}: ${error}`);
      }
      xhr.send();
    };
    if (this.checkpointManifest === null) {
      return new Promise((resolve, reject) => {
        this.loadManifest().then(() => {
          new Promise(variableRequestPromiseMethod).then(resolve);
        });
      });
    }
    return new Promise(variableRequestPromiseMethod);
  }
}

class TransformNet {
  constructor() {
    this.variables = {};
    this.variableDictionary = {};
    this.timesScalar = tf.scalar(150);
    this.plusScalar = tf.scalar(255/2);
    this.epsilonScalar = tf.scalar(1e-3);
  }

  setStyle(style) {
    this.style = style;
  }

  async load() {
    if (!this.variableDictionary.hasOwnProperty(this.style)) {
      const checkpointLoader = new CheckpointLoader("ckpts/" + this.style + "/");
      this.variableDictionary[this.style] = await checkpointLoader.getAllVariables();
    }
    this.variables = this.variableDictionary[this.style];
  }

  // TODO: use tfjs layers
  predict(preprocessedInput) {
    console.log("predict");
    const img = tf.tidy(() => {
      console.log("0");
      const conv1 = this.convLayer(preprocessedInput, 1, true, 0);
      console.log("1");
      const conv2 = this.convLayer(conv1, 2, true, 3);
      console.log("2");
      const conv3 = this.convLayer(conv2, 2, true, 6);
      console.log("3");
      const resid1 = this.residualBlock(conv3, 9);
      console.log("4");
      const resid2 = this.residualBlock(resid1, 15);
      console.log("5");
      const resid3 = this.residualBlock(resid2, 21);
      console.log("6");
      const resid4 = this.residualBlock(resid3, 27);
      console.log("7");
      const resid5 = this.residualBlock(resid4, 33);
      console.log("8");
      const convT1 = this.convTransposeLayer(resid5, 64, 2, 39);
      console.log("9");
      const convT2 = this.convTransposeLayer(convT1, 32, 2, 42);
      console.log("10");
      const convT3 = this.convLayer(convT2, 1, false, 45);
      console.log("11");
      const outTanh = tf.tanh(convT3);
      const scaled = tf.mul(this.timesScalar, outTanh);
      const shifted = tf.add(this.plusScalar, scaled);
      const clamped = tf.clipByValue(shifted, 0, 255);
      const normalized = tf.div(clamped, tf.scalar(255));
      return normalized;
    });
    return img;
  }

  convLayer(input, strides, relu, varId) {
    // console.log("convLayer");
    const y = tf.conv2d(input, this.variables[this.varName(varId)], [strides, strides], "same");
    // console.log(y);
    const y2 = this.instanceNorm(y, varId + 1);
    if (relu) {
      return tf.relu(y2);
    }
    return y2;
  }

  convTransposeLayer(input, numFilters, strides, varId) {
    const [height, width, ] = input.shape;
    const newRows = height * strides;
    const newCols = width * strides;
    const newShape = [newRows, newCols, numFilters];
    const y = tf.conv2dTranspose(input, this.variables[this.varName(varId)], newShape, [strides, strides], "same");
    const y2 = this.instanceNorm(y, varId + 1);
    const y3 = tf.relu(y2);
    return y3;
  }

  residualBlock(input, varId) {
    const conv1 = this.convLayer(input, 1, true, varId);
    const conv2 = this.convLayer(conv1, 1, false, varId + 3);
    return tf.addStrict(conv2, input);
  }

  instanceNorm(input, varId) {
    const [height, width, inDepth] = input.shape;
    const moments = tf.moments(input, [0, 1]);
    const mu = moments.mean;
    const sigmaSq = moments.variance;
    const shift = this.variables[this.varName(varId)];
    const scale = this.variables[this.varName(varId + 1)];
    const epsilon = this.epsilonScalar;
    const normalized = input.sub(mu).div(sigmaSq.add(epsilon).sqrt());
    const shifted = scale.mul(normalized).add(shift);
    return shifted.as3D(height, width, inDepth);
  }

  varName(varId) {
    if (varId === 0) {
      return "Variable";
    }
    return "Variable_" + varId;
  }

  dispose() {
    for (const styleName in this.variableDictionary) {
      for (const varName in this.variableDictionary[styleName]) {
        this.variableDictionary[styleName][varName].dispose();
      }
    }
  }
}

var transformNet = new TransformNet();

function runInference(preprocessed) {
  tf.tidy(() => {
    const converted = preprocessed.asType("float32");
    const inferenceResult = transformNet.predict(converted);
    console.log(inferenceResult);
    const imageData = new ImageData(150, 100);
    const data = inferenceResult.dataSync();
    // maybe const data = await a.data();
    const [height, width, ] = inferenceResult.shape;
    for (let i = 0; i < height * width; i++) {
      const j = i * 4;
      const k = i * 3;
      imageData.data[j + 0] = Math.round(255 * data[k + 0]);
      imageData.data[j + 1] = Math.round(255 * data[k + 1]);
      imageData.data[j + 2] = Math.round(255 * data[k + 2]);
      imageData.data[j + 3] = 255;
    }
    postMessage(imageData);
  });
}

onmessage = function(e) {
  //tf.setBackend("webgl");
  console.log("worker onmessage " + tf.getBackend());
  if (!STYLES.includes(e.data[0])) {
    console.log("invalid style");
    return;
  }

  var image = tf.image.resizeBilinear(tf.fromPixels(e.data[1]), [100, 150]);

  // console.log(image);

  // a.matMul(b).print();

  // console.log(tf.conv2d(tf.randomNormal([10, 10, 1]), tf.tensor([1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 1, 2]), [2, 2], "same"));

  // postMessage("asdf");

  transformNet.setStyle(e.data[0]);
  transformNet.load().then(() => {
    // console.log(tf.tensor3d([[[1], [2]], [[3], [4]]]));
    runInference(image);
  });
}
