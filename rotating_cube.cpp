#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/noise.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>

// Vertex Shader Source
const GLchar* vertexSource = R"glsl(
    #version 410 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    layout(location = 2) in vec3 normal;

    out vec3 fragColor;
    out vec3 FragPos;
    out vec3 Normal;

    uniform mat4 MVP;
    uniform mat4 model;
    uniform mat3 normalMatrix;

    void main() {
        fragColor = color;
        FragPos = vec3(model * vec4(position, 1.0));
        Normal = normalMatrix * normal; // Normal transformation
        gl_Position = MVP * vec4(position, 1.0);
    }
)glsl";

// Fragment Shader Source
const GLchar* fragmentSource = R"glsl(
    #version 410 core    
    in vec3 fragColor;
    in vec3 FragPos;
    in vec3 Normal;

    out vec4 color;

    uniform vec3 lightPos;
    uniform vec3 viewPos;

    uniform vec3 lightColor;
    
    uniform float u_time;

    // Classic Perlin 3D Noise by Stefan Gustavson (https://github.com/stegu/webgl-noise)
    vec4 permute(vec4 x) {
        return mod(((x * 34.0) + 1.0) * x, 289.0);
    }

    vec4 taylorInvSqrt(vec4 r) {
        return 1.79284291400159 - 0.85373472095314 * r;
    }

    vec3 fade(vec3 t) {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }

    float cnoise(vec3 P) {
        vec3 Pi0 = floor(P); // Integer part for indexing
        vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
        Pi0 = mod(Pi0, 289.0);
        Pi1 = mod(Pi1, 289.0);
        vec3 Pf0 = fract(P); // Fractional part for interpolation
        vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
        vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
        vec4 iy = vec4(Pi0.yy, Pi1.yy);
        vec4 iz0 = Pi0.zzzz;
        vec4 iz1 = Pi1.zzzz;

        vec4 ixy = permute(permute(ix) + iy);
        vec4 ixy0 = permute(ixy + iz0);
        vec4 ixy1 = permute(ixy + iz1);

        vec4 gx0 = ixy0 / 7.0;
        vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
        gx0 = fract(gx0);
        vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
        vec4 sz0 = step(gz0, vec4(0.0));
        gx0 -= sz0 * (step(0.0, gx0) - 0.5);
        gy0 -= sz0 * (step(0.0, gy0) - 0.5);

        vec4 gx1 = ixy1 / 7.0;
        vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
        gx1 = fract(gx1);
        vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
        vec4 sz1 = step(gz1, vec4(0.0));
        gx1 -= sz1 * (step(0.0, gx1) - 0.5);
        gy1 -= sz1 * (step(0.0, gy1) - 0.5);

        vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
        vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
        vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
        vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
        vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
        vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
        vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
        vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

        vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
        g000 *= norm0.x;
        g010 *= norm0.y;
        g100 *= norm0.z;
        g110 *= norm0.w;
        vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
        g001 *= norm1.x;
        g011 *= norm1.y;
        g101 *= norm1.z;
        g111 *= norm1.w;

        float n000 = dot(g000, Pf0);
        float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
        float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
        float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
        float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
        float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
        float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
        float n111 = dot(g111, Pf1);

        vec3 fade_xyz = fade(Pf0);
        vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
        vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
        float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
        return 2.2 * n_xyz;
    }

    void main() {
        // Bump mapping using Perlin noise
        float bumpHeight = 0.05;
        float n = cnoise(FragPos * 180.0 + u_time);
        vec3 bumpedNormal = Normal + n * bumpHeight;

        // Ambient lighting
        float ambientStrength = 0.15;
        vec3 ambient = ambientStrength * lightColor;
        
        // Diffuse lighting
        float diffStrength = 0.8;
        vec3 norm = normalize(bumpedNormal); // normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diffStrength * diff * lightColor;

        // Specular lighting
        float specularStrength = 0.9;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
        vec3 specular = specularStrength * spec * lightColor;

        // Compute object color
        float augR = cos(FragPos.x + u_time * 7.2315 * FragPos.z + sin(u_time) * 0.421);
        float augG = sin(FragPos.z * 10 * cos(FragPos.x * 5.3) - FragPos.x - u_time * 2.321);
        float augB = sin(u_time * 0.23 + FragPos.x * 1.2) * cos(u_time * FragPos.y * 4.53) * sin(FragPos.z * 2.53);
        vec3 objectColor = fragColor + 0.08 * vec3(augR, augG, augB);

        vec3 result = (ambient + diffuse) * objectColor + specular;
        color = vec4(result, 1.0);
    }
)glsl";

// Error checking utility
void checkSDLError(int line = -1) {
    std::string error = SDL_GetError();
    if (error != "") {
        std::cerr << "SDL Error: " << error << std::endl;
        if (line != -1) {
            std::cerr << "\nLine: " << line << std::endl;
        }
        SDL_ClearError();
    }
}

void checkShaderCompilation(GLuint shader) {
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        std::cerr << "Shader Compilation Error: " << buffer << std::endl;
    }
}

void checkProgramLinking(GLuint program) {
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, nullptr, buffer);
        std::cerr << "Program Linking Error: " << buffer << std::endl;
    }
}

GLuint createShaderProgram(const GLchar* vertexSource, const GLchar* fragmentSource) {
    // Create and compile the vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShaderCompilation(vertexShader);

    // Create and compile the fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShaderCompilation(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkProgramLinking(shaderProgram);

    // Cleanup shaders (they are linked into the program and no longer needed)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Set OpenGL attributes
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    // Create an SDL window
    SDL_Window* window = SDL_CreateWindow("OpenGL 3D Cube", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Create an OpenGL context
    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        std::cerr << "Failed to create OpenGL context: " << SDL_GetError() << std::endl;
        return -1;
    }

    std::cout << "OpenGL version supported by this platform: " << glGetString(GL_VERSION) << std::endl;

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Vertices of a cube with colors and normals
    GLfloat vertices[] = {
        // Positions          // Colors            // Normals
        -0.5f, -0.5f, -0.5f,  0.8f, 0.2f, 0.2f,   0.0f,  0.0f, -1.0f, // Red
        0.5f, -0.5f, -0.5f,  0.8f, 0.2f, 0.2f,   0.0f,  0.0f, -1.0f, // Red
        0.5f,  0.5f, -0.5f,  0.8f, 0.2f, 0.2f,   0.0f,  0.0f, -1.0f, // Red
        -0.5f,  0.5f, -0.5f,  0.8f, 0.2f, 0.2f,   0.0f,  0.0f, -1.0f, // Red

        -0.5f, -0.5f,  0.5f,  0.2f, 0.8f, 0.2f,   0.0f,  0.0f,  1.0f, // Green
        0.5f, -0.5f,  0.5f,  0.2f, 0.8f, 0.2f,   0.0f,  0.0f,  1.0f, // Green
        0.5f,  0.5f,  0.5f,  0.2f, 0.8f, 0.2f,   0.0f,  0.0f,  1.0f, // Green
        -0.5f,  0.5f,  0.5f,  0.2f, 0.8f, 0.2f,   0.0f,  0.0f,  1.0f, // Green

        -0.5f, -0.5f, -0.5f,  0.2f, 0.2f, 0.8,  -1.0f,  0.0f,  0.0f, // Blue
        -0.5f,  0.5f, -0.5f,  0.2f, 0.2f, 0.8f,  -1.0f,  0.0f,  0.0f, // Blue
        -0.5f,  0.5f,  0.5f,  0.2f, 0.2f, 0.8f,  -1.0f,  0.0f,  0.0f, // Blue
        -0.5f, -0.5f,  0.5f,  0.2f, 0.2f, 0.8f,  -1.0f,  0.0f,  0.0f, // Blue

        0.5f, -0.5f, -0.5f,  0.8f, 0.8f, 0.2f,   1.0f,  0.0f,  0.0f, // Yellow
        0.5f,  0.5f, -0.5f,  0.8f, 0.8f, 0.2f,   1.0f,  0.0f,  0.0f, // Yellow
        0.5f,  0.5f,  0.5f,  0.8f, 0.8f, 0.2f,   1.0f,  0.0f,  0.0f, // Yellow
        0.5f, -0.5f,  0.5f,  0.8f, 0.8f, 0.2f,   1.0f,  0.0f,  0.0f, // Yellow

        -0.5f, -0.5f, -0.5f,  0.8f, 0.2f, 0.8f,   0.0f, -1.0f,  0.0f, // Magenta
        0.5f, -0.5f, -0.5f,  0.8f, 0.2f, 0.8f,   0.0f, -1.0f,  0.0f, // Magenta
        0.5f, -0.5f,  0.5f,  0.8f, 0.2f, 0.8f,   0.0f, -1.0f,  0.0f, // Magenta
        -0.5f, -0.5f,  0.5f,  0.8f, 0.2f, 0.8f,   0.0f, -1.0f,  0.0f, // Magenta

        -0.5f,  0.5f, -0.5f,  0.2f, 0.8f, 0.8f,   0.0f,  1.0f,  0.0f, // Cyan
        0.5f,  0.5f, -0.5f,  0.2f, 0.8f, 0.8f,   0.0f,  1.0f,  0.0f, // Cyan
        0.5f,  0.5f,  0.5f,  0.2f, 0.8f, 0.8f,   0.0f,  1.0f,  0.0f, // Cyan
        -0.5f,  0.5f,  0.5f,  0.2f, 0.8f, 0.8f,   0.0f,  1.0f,  0.0f  // Cyan
    };

    GLuint indices[] = {
        0, 1, 2, 2, 3, 0,    // Back face (red)
        4, 5, 6, 6, 7, 4,    // Front face (green)
        8, 9, 10, 10, 11, 8, // Left face (blue)
        12, 13, 14, 14, 15, 12, // Right face (yellow)
        16, 17, 18, 18, 19, 16, // Bottom face (magenta)
        20, 21, 22, 22, 23, 20  // Top face (cyan)
    };

    // Create Vertex Array Object and Vertex Buffer Object
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Bind VAO
    glBindVertexArray(VAO);

    // Bind VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Bind EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Vertex Attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    // Unbind VAO
    glBindVertexArray(0);

    // Create shader programs
    GLuint shaderProgram = createShaderProgram(vertexSource, fragmentSource);

    // Define the MVP matrix
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(1.5f, 1.5f, 1.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 10.0f);
    glm::mat4 mvp = projection * view * model;
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));

    glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
    glm::vec3 viewPos(0.0f, 0.0f, 3.0f);

    // Get the locations of the uniforms
    GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    GLint lightColorLoc = glGetUniformLocation(shaderProgram, "lightColor");
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint normalMatrixLoc = glGetUniformLocation(shaderProgram, "normalMatrix");

    GLint mvpLocation = glGetUniformLocation(shaderProgram, "MVP");
    GLint timeLocation = glGetUniformLocation(shaderProgram, "u_time");

    auto startTime = std::chrono::high_resolution_clock::now();

    float prevTimeValue = 0.0f;
    glm::mat4 prevRotation = glm::mat4(1.0f);

    float rotationX = 0.0f;
    float rotationY = 0.0f;
    float rotationZ = 0.0f;
    int lastMouseX, lastMouseY;
    bool firstMouse = true;

    float shakeAmplitude = 0.0;

    // Desired frame rate
    const int targetFPS = 100;
    const int frameDelay = 1000 / targetFPS;

    // Main loop
    bool running = true;
    bool autoRotation = true;

    SDL_Event event;
    while (running) {
        Uint32 frameStart = SDL_GetTicks();

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
                running = false;
            }
            else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_SPACE) {
                autoRotation = !autoRotation;
            }
            else if (event.type == SDL_MOUSEMOTION) {
                int mouseX = event.motion.x;
                int mouseY = event.motion.y;

                if (firstMouse) {
                    lastMouseX = mouseX;
                    lastMouseY = mouseY;
                    firstMouse = false;
                }

                float xoffset = mouseX - lastMouseX;
                float yoffset = mouseY - lastMouseY;

                lastMouseX = mouseX;
                lastMouseY = mouseY;

                const float sensitivity = 0.5f;
                xoffset *= sensitivity;
                yoffset *= sensitivity;

                rotationX += yoffset;
                rotationY += xoffset;
            }
        }

        // Calculate the elapsed time
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = currentTime - startTime;
        float timeValue = elapsed.count(); // Get the elapsed time in seconds

        // Rotate the cube based on mouse input and auto-rotation
        if (autoRotation) {
            rotationX += (timeValue - prevTimeValue) * 36.5 * sin(timeValue);
            rotationY += (timeValue - prevTimeValue) * 25.0 * cos(timeValue);
            rotationZ += (timeValue - prevTimeValue) * 5.21;
        }

        glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(rotationX), glm::vec3(1.0f, 0.0f, 0.0f));
        rotation = glm::rotate(rotation, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));
        rotation = glm::rotate(rotation, glm::radians(rotationZ), glm::vec3(0.0f, 0.0f, 1.0f));

        // Compute the angular velocity
        glm::mat4 deltaRotation = rotation * glm::inverse(prevRotation);
        glm::vec3 angularVelocity = glm::eulerAngles(glm::quat_cast(deltaRotation));

        // Calculate shake amplitude based on rotation speed
        float shakeAmplitude = std::min(0.1f, std::max(0.0f, shakeAmplitude * (1.0f - 3.0f * (timeValue - prevTimeValue))) + 0.1f * glm::length(angularVelocity));
        float shakeX = shakeAmplitude * glm::perlin(glm::vec3(timeValue * 80.1f, timeValue * 30.1f, timeValue * 80.4f));
        float shakeY = shakeAmplitude * glm::perlin(glm::vec3(timeValue * 70.1f, timeValue * 20.2f, timeValue * 70.5f));
        float shakeZ = shakeAmplitude * glm::perlin(glm::vec3(timeValue * 60.1f, timeValue * 52.3f, timeValue * 90.6f));

        // Update the model matrix
        glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(shakeX, shakeY, shakeZ));
        glm::mat4 model = translation * rotation;

        glm::mat4 mvp = projection * view * model;
        glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the shader program
        glUseProgram(shaderProgram);

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix3fv(normalMatrixLoc, 1, GL_FALSE, glm::value_ptr(normalMatrix));
        glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));

        // Set light and view positions
        glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(viewPos));
        glUniform3fv(lightColorLoc, 1, glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));

        glUniform1f(timeLocation, timeValue);

        // Draw the cube
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // Swap buffers
        SDL_GL_SwapWindow(window);

        // Calculate the duration of the frame
        Uint32 frameTime = SDL_GetTicks() - frameStart;

        // If the frame took less time than the target frame delay, wait the remaining time
        if (frameDelay > frameTime) {
            SDL_Delay(frameDelay - frameTime);
        }

        // Update the previous time and rotation axis for the next frame
        prevTimeValue = timeValue;
        prevRotation = rotation;
    }

    // Cleanup
    glDeleteProgram(shaderProgram);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
