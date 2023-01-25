#include "stdafx.h"
#include "Renderer.h"

#include "ChronoUtilities.h"
#include "InputManager.h"
#include "ShaderProgramDB.h"

// Public methods

AlgGeom::Renderer::Renderer(): _appState(nullptr), _content(nullptr), _screenshoter(nullptr), _triangleShader(nullptr), _lineShader(nullptr), _pointShader(nullptr)
{
    _gui = GUI::getInstance();
}

void AlgGeom::Renderer::renderLine(Model3D::MatrixRenderInformation* matrixInformation)
{
    _lineShader->use();

    for (auto& model : _content->_model)
    {
        model->draw(_lineShader, matrixInformation, _appState, GL_LINES);
    }
}

void AlgGeom::Renderer::renderPoint(Model3D::MatrixRenderInformation* matrixInformation)
{
    _pointShader->use();

    for (auto& model : _content->_model)
    {
        model->draw(_pointShader, matrixInformation, _appState, GL_POINTS);
    }
}

void AlgGeom::Renderer::renderTriangle(Model3D::MatrixRenderInformation* matrixInformation)
{
    _triangleShader->use();
    this->transferLightUniforms(_triangleShader);
    _triangleShader->setUniform("gamma", _appState->_gamma);

    for (auto& model : _content->_model)
    {
        model->draw(_triangleShader, matrixInformation, _appState, GL_TRIANGLES);
    }
}

void AlgGeom::Renderer::transferLightUniforms(RenderingShader* shader)
{
    shader->setUniform("lightPosition", _appState->_lightPosition);
    shader->setUniform("Ia", _appState->_Ia);
    shader->setUniform("Id", _appState->_Id);
    shader->setUniform("Is", _appState->_Is);
}

// Private methods

AlgGeom::Renderer::~Renderer()
{
    delete _screenshoter;
}

void AlgGeom::Renderer::createCamera()
{
    if (_content->_model.size())
    {
        _content->_camera[_appState->_selectedCamera]->track(_content->_model[0].get());
        _content->_camera[_appState->_selectedCamera]->saveCamera();
    }
}

void AlgGeom::Renderer::createModels()
{
    _content->buildScenario();
}

void AlgGeom::Renderer::createShaderProgram()
{
    _pointShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::POINT_RENDERING);
    _lineShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::LINE_RENDERING);
    _triangleShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::TRIANGLE_RENDERING);
}

void AlgGeom::Renderer::prepareOpenGL(uint16_t width, uint16_t height, ApplicationState* appState)
{
    _appState = appState;
    _appState->_viewportSize = ivec2(width, height);
    _content = new SceneContent{};
    _screenshoter = new FBOScreenshot(width, height);

    // - Establecemos un gris medio como color con el que se borrará el frame buffer.
    // No tiene por qué ejecutarse en cada paso por el ciclo de eventos.
    glClearColor(_appState->_backgroundColor.x, _appState->_backgroundColor.y, _appState->_backgroundColor.z, 1.0f);

    // - Le decimos a OpenGL que tenga en cuenta la profundidad a la hora de dibujar.
    // No tiene por qué ejecutarse en cada paso por el ciclo de eventos.
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_MULTISAMPLE);

    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(RESTART_PRIMITIVE_INDEX);

    glEnable(GL_PROGRAM_POINT_SIZE);

    glEnable(GL_POLYGON_OFFSET_FILL);

    _content->_camera.push_back(std::unique_ptr<Camera>(new Camera(width, height)));
    this->createShaderProgram();
    this->createModels();
    this->createCamera();

    // Observer
    InputManager* inputManager = InputManager::getInstance();
    inputManager->suscribeResize(this);
    inputManager->suscribeScreenshot(this);

    this->resizeEvent(_appState->_viewportSize.x, _appState->_viewportSize.y);
}

void AlgGeom::Renderer::removeModel()
{
    if (!_content->_model.empty())
        _content->_model.erase(_content->_model.end() - 1);
}

void AlgGeom::Renderer::resizeEvent(uint16_t width, uint16_t height)
{
    glViewport(0, 0, width, height);

    _appState->_viewportSize = ivec2(width, height);
    _content->_camera[_appState->_selectedCamera]->setRaspect(width, height);
}

void AlgGeom::Renderer::screenshotEvent(const ScreenshotEvent& event)
{
    if (event._type == ScreenshotListener::RGBA)
    {
        const ivec2 size = _appState->_viewportSize;
        const ivec2 newSize = ivec2(_appState->_viewportSize.x * _appState->_screenshotFactor, _appState->_viewportSize.y * _appState->_screenshotFactor);

        this->resizeEvent(newSize.x, newSize.y);
        this->render(.0f, false, true);
        _screenshoter->saveImage(_appState->_screenshotFilenameBuffer);

        this->resizeEvent(size.x, size.y);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void AlgGeom::Renderer::render(float alpha, bool renderGui, bool bindScreenshoter)
{
    Model3D::MatrixRenderInformation matrixInformation;
    glm::mat4 bias = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f));

    matrixInformation.setMatrix(Model3D::MatrixRenderInformation::VIEW, _content->_camera[0]->getViewMatrix());
    matrixInformation.setMatrix(Model3D::MatrixRenderInformation::VIEW_PROJECTION, _content->_camera[0]->getViewProjectionMatrix());

    if (bindScreenshoter)
    {
        _screenshoter->modifySize(_appState->_viewportSize.x, _appState->_viewportSize.y);
        _screenshoter->bindFBO();
    }

    glClearColor(_appState->_backgroundColor.x, _appState->_backgroundColor.y, _appState->_backgroundColor.z, alpha);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPolygonOffset(1.0f, 1.0f);

    if (_appState->_activeRendering[VAO::IBO_TRIANGLE])
    {
        this->renderTriangle(&matrixInformation);
    }

    if (_appState->_activeRendering[VAO::IBO_LINE])
    {
        this->renderLine(&matrixInformation);
    }

    if (_appState->_activeRendering[VAO::IBO_POINT])
    {
        this->renderPoint(&matrixInformation);
    }

    glPolygonOffset(.0f, .0f);

    if (renderGui)
        _gui->render(_content);

    _appState->_numFps = _gui->getFrameRate();
}
