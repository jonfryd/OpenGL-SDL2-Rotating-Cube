# OpenGL-SDL2-Rotating-Cube

A simple OpenGL and SDL2 project that renders a 3D rotating cube with specular lighting. Each face of the cube is uniquely colored, and the cube rotates continuously, demonstrating basic 3D graphics techniques including transformations, lighting, and shading.

## Prerequisites

- **SDL2**: Simple DirectMedia Layer library for handling window creation and input.
- **GLEW**: OpenGL Extension Wrangler Library to manage OpenGL extensions.
- **GLM**: OpenGL Mathematics library for handling matrix and vector operations.

## Installation

1. **Install Dependencies**:
    - On macOS:
        ```sh
        brew install sdl2 glew glm
        ```
    - On Linux (Ubuntu):
        ```sh
        sudo apt-get install libsdl2-dev libglew-dev libglm-dev
        ```

2. **Clone the Repository**:
    ```sh
    git clone https://github.com/jonfryd/OpenGL-SDL2-Rotating-Cube.git
    cd OpenGL-SDL2-Rotating-Cube
    ```

## Building the Project on macOS

1. **Compile the Project in a Homebrew environment**:
    ```sh
    g++ -o rotating_cube rotating_cube.cpp -I/opt/homebrew/include/ -L/opt/homebrew/lib/ -lSDL2 -lGLEW -framework OpenGL -std=c++11
    ```

2. **Run the Executable**:
    ```sh
    ./rotating_cube
    ```

## Usage

Run the compiled executable to see the continuously rotating cube with specular lighting.
