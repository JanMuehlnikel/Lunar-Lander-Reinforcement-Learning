# Lunar-Lander-Reinforcement-Learning

This project has been implemented with the gymnasium Framework: [https://gymnasium.farama.org/environments/atari/freeway/](https://gymnasium.farama.org/environments/box2d/lunar_lander/)


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Side by Side Images</title>
    <style>
        .container {
            display: flex;
            justify-content: center;
            gap: 10px; /* Adjust space between images */
        }
        .container img {
            border: 1px solid #ccc; /* Optional: adds a border around images */
        }
    </style>
</head>
<body>
    <div class="container">
        <img src="data/images/ep_100.gif" alt="Successful Try" width="600" height="400">
        <img src="data/images/final.gif" alt="Bad Try" width="600" height="400">
    </div>
</body>
</html>

## Try after few episodes:
<img src="data/images/ep_100.gif" alt="Successful Try" width="600" height="400">
<br>

## Trained Agent
<img src="data/images/final.gif" alt="Bad Try" width="600" height="400">
<br>

# Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone [https://github.com/JanMuehlnikel/Atari-Freeway-Reinforcement-Learning](https://github.com/JanMuehlnikel/Lunar-Lander-Reinforcement-Learning)
    cd your-repo
    ```

2. **Create and Activate a New Conda Environment**:
    ```bash
    conda create --name LunarEnv python=3.10
    conda activate LunarEnv

3. **Install `pip` in the New Conda Environment**:
    ```bash
    conda install pip
    ```

4. **Install Packages from `requirements.txt`**:
    ```bash
    pip install -r requirements.txt
    ```
