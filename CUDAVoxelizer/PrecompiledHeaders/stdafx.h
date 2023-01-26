#include <any>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits.h>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// IMPORTANTE: El include de GLEW debe estar siempre ANTES de el de GLFW
#include "GL/glew.h"								
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext/vector_relational.hpp"  
#include "glm/gtc/epsilon.hpp"   
#include "glm/gtx/norm.hpp"

// Gui
#include "imgui.h"
#include "imgui/examples/imgui_impl_glfw.h"
#include "imgui/examples/imgui_impl_opengl3.h"

// CUDA
#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Rename
typedef glm::vec4 vec4;
typedef glm::mat4 mat4;

typedef glm::vec3 vec3;
typedef glm::mat3 mat3;

typedef glm::vec2 vec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec2 uvec2;

typedef glm::ivec3 ivec3;
typedef glm::ivec2 ivec2;
