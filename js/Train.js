import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const logButton = document.getElementById("logButton")
const enableWebcamButton = document.getElementById("webcamButton")
const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let results = undefined;

/********************************************************************
 // EXPORT TRAINING DATA
 ********************************************************************/
function exportTrainingData() {
    let storedData = localStorage.getItem('handData');
    if (storedData) {
        let blob = new Blob([storedData], { type: 'application/json' });
        let url = URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.href = url;
        a.download = 'trainingData.json';
        a.click();
        URL.revokeObjectURL(url);
    } else {
        console.log("No training data to export.");
    }
}

/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")
    if (hasGetUserMedia()) {
        enableWebcamButton.addEventListener("click", (e) => enableCam(e))
        logButton.addEventListener("click", (e) => logAllHands(e))
    }
}

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

/********************************************************************
 // START THE WEBCAM
 ********************************************************************/
function enableCam() {
    webcamRunning = true
    navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then((stream) => {
        video.srcObject = stream
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth
            canvasElement.style.height = video.videoHeight
            canvasElement.width = video.videoWidth
            canvasElement.height = video.videoHeight
            document.querySelector(".videoView").style.height = video.videoHeight + "px"
            predictWebcam()
        })
    })
}

/********************************************************************
 // START PREDICTIONS
 ********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for (let hand of results.landmarks) {
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}

/********************************************************************
 // LOG HAND COORDINATES IN THE CONSOLE
 ********************************************************************/
function logAllHands() {
    for (let hand of results.landmarks) {
        console.log(hand)
    }
}

createHandLandmarker()

/********************************************************************
 // NN TRAIN
 ********************************************************************/

let trainingData = [];

// Adds example data
function addExample(label) {
    if (!results || results.landmarks.length === 0) return;

    // Use only the first hand (expand to two hands if needed)
    const landmarks = results.landmarks[0];

    // Flatten all x, y, z values into one array
    const input = landmarks.flatMap(point => [point.x, point.y, point.z]);

    if (input.length === 63) {
        trainingData.push({ input, label });

        // Optionally save temporarily in localStorage
        localStorage.setItem('handData', JSON.stringify(trainingData));
        console.log(`Saved example for: ${label}`);
    } else {
        console.error("Input data does not have the correct shape.");
    }
}

// Button listeners for adding examples
document.getElementById("upButton").addEventListener("click", () => addExample("up"));
document.getElementById("downButton").addEventListener("click", () => addExample("down"));
document.getElementById("leftButton").addEventListener("click", () => addExample("left"));
document.getElementById("rightButton").addEventListener("click", () => addExample("right"));
document.getElementById("exportButton").addEventListener("click", exportTrainingData);

// Setup the neural network
const nnOptions = {
    inputs: 63,  // 21 points × 3 (x, y, z)
    outputs: 4,  // up, down, left, right
    task: 'classification',
    debug: true
};

ml5.setBackend("webgl");
const options = {
    task: 'classification',
    debug: true,
    layers: [
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            activation: 'softmax',
        },
    ]
}

const nn = ml5.neuralNetwork(options)
let train = []
let test = []

// Load data and train the model
async function loadDataAndTrain() {
    try {
        const storedData = localStorage.getItem('handData');
        if (!storedData) {
            console.error("No training data found.");
            return;
        }

        const parsedData = JSON.parse(storedData);
        parsedData.sort(() => (Math.random() - 0.5))

        train = parsedData.slice(0, Math.floor(parsedData.length * 0.8))
        test = parsedData.slice(Math.floor(parsedData.length * 0.8))

        for (let item of train) {
            nn.addData(item.input, { label: item.label });
        }

        nn.normalizeData();

        nn.train({ epochs: 50 }, () => {
            console.log("Training complete!");
        });
    } catch (error) {
        console.error("Error loading training data:", error);
    }
}

// Export trained model
function exportTraining() {
    try {
        nn.save(); // Optional: save the trained model
        console.log("Model saved.");
    } catch (error) {
        console.error("No model trained.");
    }
}

document.getElementById("TrainButton").addEventListener("click", loadDataAndTrain);
document.getElementById("NNButton").addEventListener("click", exportTraining);

// Testing accuracy
document.getElementById("AccuracyButton").addEventListener("click", testing);

async function testing() {
    let correct = 0;

    for (const { input, label } of test) {
        const prediction = await nn.classify(input);
        if (prediction[0].label === label) {
            correct++;
        }
    }
    const accuracy = (correct / test.length) * 100;
    console.log(`Accuracy: ${accuracy}%`);
}

let confusionMatrix = {
    up: { up: 0, down: 0, left: 0, right: 0 },
    down: { up: 0, down: 0, left: 0, right: 0 },
    left: { up: 0, down: 0, left: 0, right: 0 },
    right: { up: 0, down: 0, left: 0, right: 0 }
};

// Accuracy Test
document.getElementById("AccuracyButton").addEventListener("click", async () => {
    let correct = 0;
    let total = test.length;

    // Loop over the test data and check predictions
    for (const { input, label } of test) {
        const prediction = await nn.classify(input);
        const predictedLabel = prediction[0].label;

        // Increment the correct count if the prediction is correct
        if (predictedLabel === label) {
            correct++;
        }

        // Update the confusion matrix
        confusionMatrix[label][predictedLabel]++;
    }

    // Calculate accuracy
    const accuracy = (correct / total) * 100;
    document.getElementById("accuracyResult").innerHTML = `Accuracy: ${accuracy.toFixed(2)}%`;
});

// Show confusion matrix in a table format
document.getElementById("tabelButton").addEventListener("click", () => {
    displayConfusionMatrix();
});

// Function to display the confusion matrix
function displayConfusionMatrix() {
    const matrixElement = document.getElementById("confusionMatrixResult");
    let matrixHTML = "<table border='1'><tr><th>Actual/Pred</th><th>Up</th><th>Down</th><th>Left</th><th>Right</th></tr>";

    // Loop over each label (up, down, left, right)
    for (let actualLabel of ['up', 'down', 'left', 'right']) {
        matrixHTML += `<tr><td>${actualLabel}</td>`;
        // Loop over each predicted label and populate the confusion matrix
        for (let predictedLabel of ['up', 'down', 'left', 'right']) {
            matrixHTML += `<td>${confusionMatrix[actualLabel][predictedLabel]}</td>`;
        }
        matrixHTML += "</tr>";
    }
    matrixHTML += "</table>";

    matrixElement.innerHTML = matrixHTML;
}

