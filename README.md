# OpenGL-SDL2-Rotating-Cube

A simple OpenGL and SDL2 project that renders a 3D rotating cube with specular lighting. Each face of the cube is uniquely colored, and the cube rotates continuously, demonstrating basic 3D graphics techniques including transformations, lighting, and fragment shading.

https://github.com/user-attachments/assets/6003e685-5853-49a0-8ad6-8d329cae699d

## Prerequisites

- **SDL2**: Simple DirectMedia Layer library for handling window creation and input.
- **GLEW**: OpenGL Extension Wrangler Library to manage OpenGL extensions.
- **GLM**: OpenGL Mathematics library for handling matrix and vector operations.

## Installation

1. **Install Dependencies**:
    - On macOS (Homebrew):
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

## Building the Project

1. **Compile the Project**:
    Simply run `make` in the project directory. The `Makefile` will automatically detect your operating system and use the appropriate commands.

    ```sh
    make
    ```

2. **Run the Executable**:
    ```sh
    ./rotating_cube
    ```

## Usage

Run the compiled executable to see the continuously rotating cube with specular lighting.

## Controls

- Space: Toggle auto-rotation
- Escape: Quit

Move the mouse to manually apply additional rotation.
