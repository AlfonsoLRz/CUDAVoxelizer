#include "stdafx.h"
#include "Model3D.h"

#include "assimp/Exporter.hpp"
#include "../CUDAHandler.h"
#include "RandomUtilities.h"
#include "tinyply.h"
#include "../Voxelization.cuh"

// Static properties

std::string AlgGeom::Model3D::CHECKER_PATTERN_PATH = "Assets/Textures/Checker.png";
std::unordered_set<std::string> AlgGeom::Model3D::USED_NAMES;

// Public methods

AlgGeom::Model3D::Model3D(): _modelMatrix(1.0f), _voxelization(nullptr)
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
    shader->setSubroutineUniform(GL_VERTEX_SHADER, "instanceUniform", "singleInstanceUniform");

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

    if (_voxelization)
    {
        _voxelization->setModelMatrix(this->_modelMatrix);
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

AlgGeom::DrawVoxelization* AlgGeom::Model3D::voxelize(const uvec3& voxelizationDimensions)
{
    // Define voxels
    size_t numThreadsBlock = 512;
    size_t numSamples = 1024;
    size_t numVoxels = voxelizationDimensions.x * voxelizationDimensions.y * voxelizationDimensions.z;

    int* voxels = (int*)calloc(numVoxels, sizeof(int)), *voxelGPU;
    vec2* noise = (vec2*)malloc(numSamples * sizeof(vec2)), * noiseGPU;
    VAO::Vertex* verticesGPU;
    unsigned* indicesGPU;
    size_t occupiedVoxels = 0; size_t* occupiedVoxelsGPU;

    cudaEvent_t startTimer = 0, stopTimer = 0;
    CUDAHandler::startTimer(startTimer, stopTimer);

    // Fill noise buffer
    for (int sample = 0; sample < numSamples; ++sample)
        noise[sample] = vec2(RandomUtilities::getUniformRandom(), RandomUtilities::getUniformRandom());

    CUDAHandler::initializeBufferGPU(voxelGPU, numVoxels, voxels);
    CUDAHandler::initializeBufferGPU(noiseGPU, numSamples, noise);

    AABBGPU aabb (this->getAABB(false), voxelizationDimensions);
    CUDAHandler::checkError(cudaMemcpyToSymbol(c_aabb, &aabb, sizeof(AABBGPU)));
    CUDAHandler::checkError(cudaMemcpyToSymbol(c_gridDimensions, &voxelizationDimensions, sizeof(uvec3)));

    // Create vertices and indices buffer
    size_t numVertices = 0, numIndices = 0, currentVertices = 0, currentIndices = 0;
    for (auto& component : _components)
    {
        numVertices += component->_vertices.size();
        numIndices += component->_indices[VAO::IBO_TRIANGLE].size();
    }

    CUDAHandler::initializeBufferGPU(verticesGPU, numVertices);
    CUDAHandler::initializeBufferGPU(indicesGPU, numIndices);

    // Overlap execution
    std::vector<cudaStream_t> dataStream (_components.size());
    for (auto& dataStreamId : dataStream)
    {
        CUDAHandler::checkError(cudaStreamCreate(&dataStreamId));
    }
     
    // Fill voxels
    for (int componentId = 0; componentId < _components.size(); ++componentId)
    {
        std::vector<VAO::Vertex>* vertices = &_components[componentId]->_vertices;
        std::vector<GLuint>* indices = &_components[componentId]->_indices[VAO::IBO_TRIANGLE];
        size_t numFaces = indices->size() / 4;

        CUDAHandler::checkError(cudaMemcpyAsync(&verticesGPU[currentVertices], vertices->data(), vertices->size() * sizeof(VAO::Vertex), cudaMemcpyHostToDevice, dataStream[componentId]));
        CUDAHandler::checkError(cudaMemcpyAsync(&indicesGPU[currentIndices], indices->data(), indices->size() * sizeof(unsigned), cudaMemcpyHostToDevice, dataStream[componentId]));
        voxelizeComponent<<<CUDAHandler::getNumBlocks(numFaces * numSamples, numThreadsBlock), numThreadsBlock, 0, dataStream[componentId]>>>(numFaces, currentVertices, currentIndices, numSamples, voxelGPU, verticesGPU, indicesGPU, noiseGPU);

        currentVertices += vertices->size();
        currentIndices += indices->size();
    }

    // Count occupied voxels
    {
        CUDAHandler::initializeBufferGPU(occupiedVoxelsGPU, 1, &occupiedVoxels);

        countOccupiedVoxels<<<CUDAHandler::getNumBlocks(numVoxels, numThreadsBlock), numThreadsBlock>>>(numVoxels, voxelGPU, occupiedVoxelsGPU);

        CUDAHandler::downloadBufferGPU(occupiedVoxelsGPU, &occupiedVoxels, 1);
    }

    // Generate voxel' translations
    {
        size_t numTranslationVectors = occupiedVoxels;
        occupiedVoxels = 0;
        vec3* translationBuffer = (vec3*)malloc(numTranslationVectors * sizeof(vec3)), * translationGPU;

        CUDAHandler::initializeBufferGPU(translationGPU, numTranslationVectors);
        CUDAHandler::initializeBufferGPU(occupiedVoxelsGPU, 1, &occupiedVoxels);

        generateVoxelTranslation<<<CUDAHandler::getNumBlocks(numVoxels, numThreadsBlock), numThreadsBlock>>>(numVoxels, voxelGPU, occupiedVoxelsGPU, translationGPU);

        CUDAHandler::downloadBufferGPU(translationGPU, translationBuffer, numTranslationVectors);

        printf("Response time (ms): %f\n", CUDAHandler::stopTimer(startTimer, stopTimer));

        _voxelization = new DrawVoxelization();
        _voxelization->loadVoxelization(translationBuffer, numTranslationVectors, aabb.getStepLength());
        this->writeVoxelizationObj("voxels.obj", translationBuffer, aabb.getStepLength(), numTranslationVectors);

        free(translationBuffer);
    }

    CUDAHandler::free(voxelGPU);
    CUDAHandler::free(verticesGPU);
    CUDAHandler::free(indicesGPU);
    free(noise);
    free(voxels);

    // Destroy data streams
    for (auto& dataStreamId : dataStream)
    {
        CUDAHandler::checkError(cudaStreamDestroy(dataStreamId));
    }

    return _voxelization;
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

void AlgGeom::Model3D::writeVoxelizationObj(const std::string& path, vec3* translationVectors, vec3 scale, size_t numVoxels)
{
    Component* voxel = DrawVoxelization::getVoxel();
    size_t numVertices = voxel->_vertices.size();
    size_t numFaces = voxel->_indices[VAO::IBO_TRIANGLE].size() / 4;
    aiVector3D* position = new aiVector3D[numVertices * numVoxels], * normal = new aiVector3D[numVertices * numVoxels];
    aiFace* indices = new aiFace[numVoxels * numFaces];

    for (int voxelId = 0; voxelId < numVoxels; ++voxelId)
    {
        unsigned startIndex = voxelId * numVertices;

        #pragma omp parallel for
        for (int v = 0; v < numVertices; ++v)
        {
            vec3 tPosition = voxel->_vertices[v]._position * scale + translationVectors[voxelId];
            position[v + voxelId * numVertices] = aiVector3D{ tPosition.x, tPosition.y, tPosition.z };
            normal[v + voxelId * numVertices] = aiVector3D{ voxel->_vertices[v]._normal.x, voxel->_vertices[v]._normal.y, voxel->_vertices[v]._normal.z };
        }

        #pragma omp parallel for
        for (int i = 0; i < voxel->_indices[VAO::IBO_TRIANGLE].size(); i += 4)
        {
            indices[voxelId * numFaces + i / 4].mNumIndices = 3;
            indices[voxelId * numFaces + i / 4] .mIndices = new unsigned[3] {
                        voxel->_indices[VAO::IBO_TRIANGLE][i] + startIndex,
                        voxel->_indices[VAO::IBO_TRIANGLE][i + 1] + startIndex,
                        voxel->_indices[VAO::IBO_TRIANGLE][i + 2] + startIndex
            };
        }
    }

    aiMesh* mesh = new aiMesh();                     
    mesh->mNumVertices = numVertices * numVoxels;
    mesh->mVertices = position;
    mesh->mNormals = normal;
    mesh->mNumFaces = numFaces * numVoxels;
    mesh->mFaces = indices;
    mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE; 

    aiMaterial* material = new aiMaterial();     
    aiNode* root = new aiNode();                  
    root->mNumMeshes = 1;
    root->mMeshes = new unsigned [1] { 0 };      

    aiScene* out = new aiScene();                
    out->mNumMeshes = 1;
    out->mMeshes = new aiMesh* [1] { mesh };           
    out->mNumMaterials = 1;
    out->mMaterials = new aiMaterial * [1] { material }; 
    out->mRootNode = root;
    out->mMetaData = new aiMetadata();

    Assimp::Exporter exporter;
    if (exporter.Export(out, "objnomtl", path) != AI_SUCCESS)
        throw std::runtime_error(exporter.GetErrorString());

    delete out;
}

void AlgGeom::Model3D::writeVoxelizationPly(const std::string& path, vec3* translationVectors, vec3 scale, size_t numVoxels)
{
    std::filebuf fileBufferBinary;
    fileBufferBinary.open(path, std::ios::out | std::ios::binary);

    std::ostream outstreamBinary(&fileBufferBinary);
    if (outstreamBinary.fail()) throw std::runtime_error("Failed to open " + path);

    tinyply::PlyFile plyFile;
    std::vector<vec3> position, normal;
    std::vector<uvec3> triangleMesh;
    Component* voxel = DrawVoxelization::getVoxel();

    for (int voxelId = 0; voxelId < numVoxels; ++voxelId)
    {
        unsigned startIndex = position.size();

        for (VAO::Vertex& vertex : voxel->_vertices)
        {
            position.push_back(vertex._position * scale + translationVectors[voxelId]);
            normal.push_back(vertex._normal);
        }

        for (int i = 0; i < voxel->_indices[VAO::IBO_TRIANGLE].size(); i += 4)
        {
            triangleMesh.push_back(uvec3(voxel->_indices[VAO::IBO_TRIANGLE][i], voxel->_indices[VAO::IBO_TRIANGLE][i + 1], voxel->_indices[VAO::IBO_TRIANGLE][i + 2]) + startIndex);
        }
    }

    plyFile.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT32, position.size(), reinterpret_cast<uint8_t*>(position.data()), tinyply::Type::INVALID, 0);
    plyFile.add_properties_to_element("vertex", { "nx", "ny", "nz" }, tinyply::Type::FLOAT32, normal.size(), reinterpret_cast<uint8_t*>(normal.data()), tinyply::Type::INVALID, 0);
    plyFile.add_properties_to_element("face", { "vertex_index" }, tinyply::Type::UINT32, triangleMesh.size(), reinterpret_cast<uint8_t*>(triangleMesh.data()), tinyply::Type::UINT8, 3);
    plyFile.write(outstreamBinary, true);

    delete voxel;
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
    auto isIncluded = [&](int index1, int index2) -> bool
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


// .-. VOXELIZATION ._.

AlgGeom::DrawVoxelization::DrawVoxelization() : Model3D()
{
}

AlgGeom::DrawVoxelization::~DrawVoxelization()
{
}

void AlgGeom::DrawVoxelization::draw(RenderingShader* shader, MatrixRenderInformation* matrixInformation, ApplicationState* appState, GLuint primitive)
{
    Component* component = _components[0].get();

    shader->setSubroutineUniform(GL_VERTEX_SHADER, "instanceUniform", "multiInstanceUniform");
    shader->setUniform("globalScale", _voxelLength);

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

        if (!component->_activeRendering[rendering]) return;

        shader->setUniform("mModelViewProj", matrixInformation->multiplyMatrix(MatrixRenderInformation::VIEW_PROJECTION, this->_modelMatrix));
        shader->applyActiveSubroutines();

        component->_vao->drawObject(rendering, primitive, static_cast<GLuint>(component->_indices[rendering].size()), _numVoxels);

        matrixInformation->undoMatrix(MatrixRenderInformation::VIEW);
        matrixInformation->undoMatrix(MatrixRenderInformation::VIEW_PROJECTION);
    }
}

AlgGeom::DrawVoxelization* AlgGeom::DrawVoxelization::loadVoxelization(vec3* translation, size_t numVoxels, vec3 voxelScale)
{
    Component* component = this->getVoxel();
    _numVoxels = numVoxels;
    _voxelLength = voxelScale;

    // Define instances
    {
        this->buildVao(component);
        component->_vao->defineMultiInstancingVBO(VAO::VBO_MULTI_POSITION, vec3(.0f), .0f, GL_FLOAT);
        component->_vao->setVBOData(VAO::VBO_MULTI_POSITION, translation, numVoxels);
    }

    this->_components.push_back(std::unique_ptr<Component>(component));

    return this;
}

AlgGeom::Model3D::Component* AlgGeom::DrawVoxelization::getVoxel()
{
    Component* component = new Component;

    // Geometry
    {
        const vec3 minPosition(-.5f), maxPosition(.5f);
        const std::vector<vec3> points
        {
            vec3(minPosition[0], minPosition[1], maxPosition[2]),		vec3(maxPosition[0], minPosition[1], maxPosition[2]),
            vec3(minPosition[0], minPosition[1], minPosition[2]),	    vec3(maxPosition[0], minPosition[1], minPosition[2]),
            vec3(minPosition[0], maxPosition[1], maxPosition[2]),		vec3(maxPosition[0], maxPosition[1], maxPosition[2]),
            vec3(minPosition[0], maxPosition[1], minPosition[2]),		vec3(maxPosition[0], maxPosition[1], minPosition[2])
        };
        const std::vector<vec3> normals
        {
            glm::normalize(vec3(-0.5f, -0.5f, 0.5f)),	glm::normalize(vec3(0.5f, -0.5f, 0.5f)),
            glm::normalize(vec3(-0.5f, -0.5f, -0.5f)),	glm::normalize(vec3(0.5f, -0.5f, -0.5f)),
            glm::normalize(vec3(-0.5f, 0.5f, 0.5f)),	glm::normalize(vec3(0.5f, 0.5f, 0.5f)),
            glm::normalize(vec3(-0.5f, 0.5f, -0.5f)),	glm::normalize(vec3(0.5f, 0.5f, -0.5f))
        };
        const std::vector<vec2> textCoords{ vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f), vec2(0.0f) };

        for (int pointIdx = 0; pointIdx < points.size(); ++pointIdx)
        {
            component->_vertices.push_back(VAO::Vertex{ points[pointIdx], normals[pointIdx], textCoords[pointIdx] });
        }
    }

    // Topology
    {
        component->_indices[VAO::IBO_TRIANGLE] = std::vector<GLuint>
        {
            0, 1, 2, RESTART_PRIMITIVE_INDEX, 1, 3, 2, RESTART_PRIMITIVE_INDEX, 4, 5, 6, RESTART_PRIMITIVE_INDEX,
            5, 7, 6, RESTART_PRIMITIVE_INDEX, 0, 1, 4, RESTART_PRIMITIVE_INDEX, 1, 5, 4, RESTART_PRIMITIVE_INDEX,
            2, 0, 4, RESTART_PRIMITIVE_INDEX, 2, 4, 6, RESTART_PRIMITIVE_INDEX, 1, 3, 5, RESTART_PRIMITIVE_INDEX,
            3, 7, 5, RESTART_PRIMITIVE_INDEX, 3, 2, 6, RESTART_PRIMITIVE_INDEX, 3, 6, 7, RESTART_PRIMITIVE_INDEX
        };

        component->generatePointCloud();
        component->generateWireframe();
    }

    return component;
}