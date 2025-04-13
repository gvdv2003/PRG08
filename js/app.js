import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// HTML elements
const enableWebcamButton = document.getElementById("webcamButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d", { willReadFrequently: true });
const restartButton = document.getElementById("restartButton");  // Speel opnieuw knop
const popup = document.getElementById("gameOverPopup");  // Game over popup
const scoreDisplay = document.getElementById("scoreDisplay"); // Voor het tonen van de score in de pop-up

let lives = 5;

const updateLives = () => {
    document.getElementById("livesBoard").innerText = `Lives: ${lives}`;
};

// ML
let handLandmarker, results;
ml5.setBackend("webgl");
const nn = ml5.neuralNetwork({ task: "classification", debug: true });
const modelDetails = {
    model: "model/model.json",
    metadata: "model/model_meta.json",
    weights: "model/model.weights.bin",
};
nn.load(modelDetails, () => console.log("Model geladen!"));

// Tekenen en utils
const drawUtils = new DrawingUtils(canvasCtx);
let webcamRunning = false;
let currentPrediction = null;

// Vijand logica
const enemies = [];
const directions = ["up", "down", "left", "right"];
let score = 0;

let enemySpeed = 0.25;  // Start snelheid
const speedIncreaseFactor = 0.005;
const maxEnemySpeed = 0.5;

const cameraBox = {
    x: () => canvasElement.width / 2 - 150,
    y: () => canvasElement.height / 2 - 150,
    width: 300,
    height: 300,
};

function spawnEnemy() {
    const dir = directions[Math.floor(Math.random() * directions.length)];
    const enemy = { direction: dir, x: 0, y: 0, speed: enemySpeed };

    switch (dir) {
        case "up":
            enemy.x = canvasElement.width / 2;
            enemy.y = -20;
            break;
        case "down":
            enemy.x = canvasElement.width / 2;
            enemy.y = canvasElement.height + 20;
            break;
        case "left":
            enemy.x = -20;
            enemy.y = canvasElement.height / 2;
            break;
        case "right":
            enemy.x = canvasElement.width + 20;
            enemy.y = canvasElement.height / 2;
            break;
    }

    enemies.push(enemy);

    // Verhoog snelheid alleen voor toekomstige vijanden, niet voor bestaande
    if (enemySpeed < maxEnemySpeed) {
        enemySpeed = Math.min(maxEnemySpeed, enemySpeed + speedIncreaseFactor);
    }
}

function drawEnemies() {
    const centerX = canvasElement.width / 2;
    const centerY = canvasElement.height / 2;

    for (let i = enemies.length - 1; i >= 0; i--) {
        const enemy = enemies[i];
        const dx = centerX - enemy.x;
        const dy = centerY - enemy.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const moveX = (dx / dist) * enemy.speed;
        const moveY = (dy / dist) * enemy.speed;

        enemy.x += moveX;
        enemy.y += moveY;

        canvasCtx.fillStyle = "red";
        canvasCtx.beginPath();
        canvasCtx.arc(enemy.x, enemy.y, 20, 0, 2 * Math.PI);
        canvasCtx.fill();

        // Als de vijand het gebied niet verlaat, verlies een leven
        if (
            enemy.x > cameraBox.x() &&
            enemy.x < cameraBox.x() + cameraBox.width &&
            enemy.y > cameraBox.y() &&
            enemy.y < cameraBox.y() + cameraBox.height
        ) {
            // Verlies een leven als vijand niet op tijd wordt uitgeschakeld
            if (lives > 0) {
                lives--;
                updateLives();
                console.log(`Leven verloren! Levens over: ${lives}`);
            }
            enemies.splice(i, 1);
        }
    }
}

function attack() {
    let adjustedPrediction = currentPrediction;
    if (adjustedPrediction === "left") adjustedPrediction = "right";
    else if (adjustedPrediction === "right") adjustedPrediction = "left";
    const index = enemies.findIndex(e => e.direction === adjustedPrediction);

    if (index !== -1) {
        enemies.splice(index, 1);
        score++;
        document.getElementById("scoreBoard").innerText = `Score: ${score}`; // ✅ Update score op het scherm
        console.log(`Vijand geraakt! Score: ${score}`);
    } else {
        console.log("Mis!");
    }

    // Als de levens op zijn, toon de game over pop-up
    if (lives <= 0) {
        showGameOver();
    }
}

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
    });
    console.log("HandLandmarker geladen");
};

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// ✅ Nieuwe vlag om interval één keer te starten
let spawnLoopStarted = false;

function enableCam() {
    webcamRunning = true;
    navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480 }, audio: false })
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                canvasElement.width = 640;
                canvasElement.height = 480;
                canvasElement.style.width = "640px";
                canvasElement.style.height = "480px";
                document.querySelector(".videoView").style.height = "480px";
                video.style.transform = "scaleX(-1)";
                canvasElement.style.transform = "scaleX(-1)";
                predictWebcam();
                gameLoop();

                // ✅ Start vijand-spawn interval als dat nog niet is gedaan
                if (!spawnLoopStarted) {
                    setInterval(spawnEnemy, 10000);
                    spawnLoopStarted = true;
                }
            });
        });
}

let lastDetection = 0;
const detectionInterval = 100;

async function predictWebcam() {
    const now = Date.now();
    if (now - lastDetection < detectionInterval) {
        requestAnimationFrame(predictWebcam);
        return;
    }

    lastDetection = now;

    if (!video || video.readyState < 2) {
        requestAnimationFrame(predictWebcam);
        return;
    }

    results = await handLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks && results.landmarks.length > 0) {
        const hand = results.landmarks[0];
        const input = hand.flatMap(point => [point.x, point.y, point.z]);

        if (input.every(coord => typeof coord === 'number' && !isNaN(coord))) {
            nn.classify(input, (results) => {
                currentPrediction = results[0].label;
            });
        }
    }

    if (webcamRunning) {
        requestAnimationFrame(predictWebcam);
    }
}

function gameLoop() {
    if (!webcamRunning) return;

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    canvasCtx.strokeStyle = "rgba(0, 255, 255, 0.5)";
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeRect(cameraBox.x(), cameraBox.y(), cameraBox.width, cameraBox.height);

    if (results?.landmarks) {
        for (let hand of results.landmarks) {
            drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
            drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
        }
    }

    drawEnemies();
    requestAnimationFrame(gameLoop);
}

document.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
        attack();
    }
});

// Functie om de game over pop-up te tonen
function showGameOver() {
    scoreDisplay.innerText = `Je eindscore is: ${score}`;
    popup.style.display = "block"; // Zorgt ervoor dat de pop-up verschijnt
}

// Event listener voor de "Speel opnieuw" knop
restartButton.addEventListener("click", () => {
    location.reload();  // Herlaadt de pagina voor een nieuw spel
});

enableWebcamButton.addEventListener("click", enableCam);
createHandLandmarker();
