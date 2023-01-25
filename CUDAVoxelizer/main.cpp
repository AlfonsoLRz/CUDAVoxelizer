#include "stdafx.h"

#include "Renderer.h"
#include "Window.h"

int main()
{
    AlgGeom::Window* window = AlgGeom::Window::getInstance();
    AlgGeom::Renderer* renderer = AlgGeom::Renderer::getInstance();

    try
    {
        window->init("Algoritmos Geometricos");
        window->loop();
    }
    catch (const std::exception& exception)
    {
        std::cout << exception.what() << std::endl;
    }

    // - Una vez terminado el ciclo de eventos, liberar recursos, etc.
    std::cout << "Finishing application..." << std::endl;

    // - Esta llamada es para impedir que la consola se cierre inmediatamente tras la
    // ejecución y poder leer los mensajes. Se puede usar también getChar();
    system("pause");
}


