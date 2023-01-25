#include "stdafx.h"
#include "Model3D.h"

#include "../CUDAHandler.h"

// Static properties

std::string AlgGeom::Model3D::CHECKER_PATTERN_PATH = "Assets/Textures/Checker.png";
std::unordered_set<std::string> AlgGeom::Model3D::USED_NAMES;

// Public methods

AlgGeom::Model3D::Model3D(): _modelMatrix(1.0f)
{
    this->overrideModelName();
}

AlgGeom::Model3D::~Model3D()
{
}

bool AlgGeom::Model3D::belongsModel(Component* component)
{
    for (auto& comp : _components)
    {
        if (comp.get() == component)
            return true;
    }

    return false;
}

void AlgGeom::Model3D::draw(RenderingShader* shader, MatrixRenderInformation* matrixInformation, ApplicationState* appState, GLuint primitive)
{
    for (auto& component : _components)
    {
        if (component->_enabled && component->_vao)
        {
            VAO::IBO_slots rendering = VAO::IBO_TRIANGLE;

            switch (primitive)
            {
            case GL_TRIANGLES:
                if (component->_material._useUniformColor)
                {
                    shader->setUniform("Kd", component->_material._kdColor);
                    shader->setSubroutineUniform(GL_FRAGMENT_SHADER, "kadUniform", "getUniformColor");
                }
                else
                {
                    Texture* checkerPattern = TextureList::getInstance()->getTexture(CHECKER_PATTERN_PATH);
                    checkerPattern->applyTexture(shader, 0, "texKdSampler");
                    shader->setSubroutineUniform(GL_FRAGMENT_SHADER, "kadUniform", "getTextureColor");
                }

                shader->setUniform("Ks", component->_material._ksColor);
                shader->setUniform("metallic", component->_material._metallic);
                shader->setUniform("roughnessK", component->_material._roughnessK);
                shader->setUniform("mModelView", matrixInformation->multiplyMatrix(MatrixRenderInformation::VIEW, this->_modelMatrix));

                break;
            case GL_LINES:
                rendering = VAO::IBO_LINE;
                shader->setUniform("lineColor", component->_material._lineColor);
                glLineWidth(component->_lineWidth);

                break;
            case GL_POINTS:
                rendering = VAO::IBO_POINT;
                shader->setUniform("pointColor", component->_material._pointColor);
                shader->setUniform("pointSize", component->_pointSize);

                break;
            }

            if (!component->_activeRendering[rendering]) continue;
            
            shader->setUniform("mModelViewProj", matrixInformation->multiplyMatrix(MatrixRenderInformation::VIEW_PROJECTION, this->_modelMatrix));
            shader->applyActiveSubroutines();

            component->_vao->drawObject(rendering, primitive, static_cast<GLuint>(component->_indices[rendering].size()));

            matrixInformation->undoMatrix(MatrixRenderInformation::VIEW);
            matrixInformation->undoMatrix(MatrixRenderInformation::VIEW_PROJECTION);
        }
    }
}

AlgGeom::Model3D* AlgGeom::Model3D::moveGeometryToOrigin(const mat4& origMatrix, float maxScale)
{
    AABB aabb = this->getAABB();

    vec3 translate = -aabb.center();
    vec3 extent = aabb.extent();
    float maxScaleAABB = std::max(extent.x, std::max(extent.y, extent.z));
    vec3 scale = (maxScale < FLT_MAX) ? ((maxScale > maxScaleAABB) ? vec3(1.0f) : vec3(maxScale / maxScaleAABB)) : vec3(1.0f);

    _modelMatrix = glm::scale(glm::mat4(1.0f), scale) * glm::translate(glm::mat4(1.0f), translate) * origMatrix;

    return this;
}

AlgGeom::Model3D* AlgGeom::Model3D::overrideModelName()
{
    std::string className = typeid(*this).name();
    std::string classTarget = "class ";
    size_t classIndex = className.find(classTarget);
    if (classIndex != std::string::npos)
    {
        className = className.substr(classIndex + classTarget.size(), className.size() - classIndex - classTarget.size());
    }

    unsigned modelIdx = 0;
    bool nameValid = false;

    while (!nameValid)
    {
        this->_name = className + " " + std::to_string(modelIdx);
        nameValid = USED_NAMES.find(this->_name) == USED_NAMES.end();
        ++modelIdx;
    }

    USED_NAMES.insert(this->_name);

    return this;
}

AlgGeom::Model3D* AlgGeom::Model3D::setLineColor(const vec3& color)
{
    for (auto& component : _components)
    {
        component->_material._lineColor = color;
    }

    return this;
}

AlgGeom::Model3D* AlgGeom::Model3D::setPointColor(const vec3& color)
{
    for (auto& component : _components)
    {
        component->_material._pointColor = color;
    }

    return this;
}

AlgGeom::Model3D* AlgGeom::Model3D::setTriangleColor(const vec4& color)
{
    for (auto& component : _components)
    {
        component->_material._kdColor = color;
    }

    return this;
}

AlgGeom::Model3D* AlgGeom::Model3D::setTopologyVisibility(VAO::IBO_slots topology, bool visible)
{
    for (auto& component : _components)
    {
        component->_activeRendering[topology] = visible;
    }

    return this;
}

void AlgGeom::Model3D::voxelize(const uvec3& voxelizationDimensions)
{
    CUDAHandler* cudaHandler = new CUDAHandler;
    cudaHandler->setDevice();

    // Define voxels
    size_t numVoxels = voxelizationDimensions.x * voxelizationDimensions.y * voxelizationDimensions.z;
    uint8_t* voxels = (uint8_t*)calloc(numVoxels, sizeof(uint8_t));
    int* bufferGPU;

    cudaHandler->initializeBufferGPU(bufferGPU, voxels, numVoxels);
    cudaHandler->sendBufferGPU(bufferGPU, voxels, numVoxels);

    cudaHandler->free(bufferGPU);
}

// Private methods

void AlgGeom::Model3D::buildVao(Component* component)
{
    VAO* vao = new VAO(true);
    vao->setVBOData(component->_vertices);
    vao->setIBOData(VAO::IBO_POINT, component->_indices[VAO::IBO_POINT]);
    vao->setIBOData(VAO::IBO_LINE, component->_indices[VAO::IBO_LINE]);
    vao->setIBOData(VAO::IBO_TRIANGLE, component->_indices[VAO::IBO_TRIANGLE]);
    component->_vao = vao;
}

void AlgGeom::Model3D::loadModelBinaryFile(const std::string& path)
{
    std::ifstream fin(path, std::ios::in | std::ios::binary);
    if (!fin.is_open())
    {
        std::cout << "Failed to open the binary file " << path << "!" << std::endl;
        return;
    }

    size_t numComponents = _components.size();
    fin.read((char*)&numComponents, sizeof(size_t));
    _components.resize(numComponents);

    for (size_t compIdx = 0; compIdx < numComponents; ++compIdx)
    {
        Component* component = new Component;
        size_t numVertices, numIndices;

        fin.read((char*)&numVertices, sizeof(size_t));
        component->_vertices.resize(numVertices);
        fin.read((char*)component->_vertices.data(), sizeof(VAO::Vertex) * numVertices);

        for (int topology = 0; topology < VAO::NUM_IBOS; ++topology)
        {
            fin.read((char*)&numIndices, sizeof(size_t));
            if (numIndices)
            {
                component->_indices[topology].resize(numIndices);
                fin.read((char*)component->_indices[topology].data(), sizeof(GLuint) * numIndices);
            }
        }

        fin.read((char*)&component->_aabb, sizeof(AABB));

        _components[compIdx] = std::unique_ptr<Component>(component);
        _aabb.update(_components[compIdx]->_aabb);
    }
}

void AlgGeom::Model3D::writeBinaryFile(const std::string& path)
{
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    if (!fout.is_open())
    {
        std::cout << "Failed to write the binary file!" << std::endl;
    }

    size_t numComponents = _components.size();
    fout.write((char*)&numComponents, sizeof(size_t));

    for (auto& component: _components)
    {
        size_t numVertices = component->_vertices.size();

        fout.write((char*)&numVertices, sizeof(size_t));
        fout.write((char*)component->_vertices.data(), numVertices * sizeof(VAO::Vertex));

        for (int topology = 0; topology < VAO::NUM_IBOS; ++topology)
        {
            size_t numIndices = component->_indices[topology].size();
            fout.write((char*)&numIndices, sizeof(size_t));
            if (numIndices) 
                fout.write((char*)component->_indices[topology].data(), numIndices * sizeof(GLuint));
        }

        fout.write((char*)(&component->_aabb), sizeof(AABB));
    }

    fout.close();
}

AlgGeom::Model3D::MatrixRenderInformation::MatrixRenderInformation()
{
    for (mat4& matrix : _matrix)
    {
        matrix = mat4(1.0f);
    }
}

void AlgGeom::Model3D::MatrixRenderInformation::undoMatrix(MatrixType type)
{
    if (_heapMatrices[type].empty())
    {
        _matrix[type] = mat4(1.0f);
    }
    else
    {
        _matrix[type] = *(--_heapMatrices[type].end());
        _heapMatrices[type].erase(--_heapMatrices[type].end());
    }
}

void AlgGeom::Model3D::Component::completeTopology()
{
    if (!this->_indices[VAO::IBO_TRIANGLE].empty())
    {
        this->generatePointCloud();
        this->generateWireframe();
    }

    if (!this->_indices[VAO::IBO_LINE].empty())
    {
        this->generatePointCloud();
    }
}

void AlgGeom::Model3D::Component::generateWireframe()
{
    std::unordered_map<int, std::unordered_set<int>> segmentIncluded;
    static auto isIncluded = [&](int index1, int index2) -> bool
    {
        std::unordered_map<int, std::unordered_set<int>>::iterator it;

        if ((it = segmentIncluded.find(index1)) != segmentIncluded.end())
        {
            if (it->second.find(index2) != it->second.end())
            {
                return true;
            }
        }

        if ((it = segmentIncluded.find(index2)) != segmentIncluded.end())
        {
            if (it->second.find(index1) != it->second.end())
            {
                return true;
            }
        }

        return false;
    };

    const size_t numIndices = this->_indices[VAO::IBO_TRIANGLE].size();

    for (size_t i = 0; i < numIndices; i += 4)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            if (!isIncluded(this->_indices[VAO::IBO_TRIANGLE][i + j], this->_indices[VAO::IBO_TRIANGLE][(j + 1) % 3 + i]))
            {
                this->_indices[VAO::IBO_LINE].push_back(this->_indices[VAO::IBO_TRIANGLE][i + j]);
                this->_indices[VAO::IBO_LINE].push_back(this->_indices[VAO::IBO_TRIANGLE][(j + 1) % 3 + i]);
                this->_indices[VAO::IBO_LINE].push_back(RESTART_PRIMITIVE_INDEX);
            }
        }
    }
}

void AlgGeom::Model3D::Component::generatePointCloud()
{
    this->_indices[VAO::IBO_POINT].resize(this->_vertices.size());
    std::iota(this->_indices[VAO::IBO_POINT].begin(), this->_indices[VAO::IBO_POINT].end(), 0);
}
