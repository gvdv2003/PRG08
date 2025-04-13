# Hand Direction Game with AI

This project is an interactive game where the player can defeat enemies by pointing in the correct direction using hand gestures. The game uses AI for hand direction detection, which is processed through a neural network. It provides a fun way to interact with the game through webcam-based hand tracking.

## Project Overview

This project uses MediaPipe for hand tracking and ML5.js for implementing a neural network to classify hand gestures. The player must point in the direction of incoming enemies (up, down, left, or right) to eliminate them. The game also tracks the player's score and lives, and displays a game-over screen when lives reach zero.

### Key Technologies Used:
- **AI**: The project uses AI to detect hand gestures and classify the direction in which the player is pointing.
- **Hand Tracking**: Utilizes MediaPipe for real-time hand tracking via the webcam.
- **Machine Learning**: Uses a custom-trained neural network to predict the hand gesture direction.
- **JavaScript/HTML5 Canvas**: For rendering the game and processing user inputs.

## Installation

To get started, follow these steps:

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/hand-direction-game.git
```

### 2. Navigate to the Project Folder

```bash
cd hand-direction-game
```

### 3. Install Dependencies


