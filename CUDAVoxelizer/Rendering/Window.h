#pragma once

#include "Renderer.h"
#include "Singleton.h"

namespace AlgGeom
{
	class Window: public Singleton<Window>
	{
		friend class Singleton<Window>;

	private:
		Renderer*	_renderer;
		GLFWwindow* _window;

	private: 
		Window();
		void releaseResources();

	public:
		virtual ~Window();
		void init(const std::string& title, const uint16_t width = 1024, const uint16_t height = 576);
		void loop();
	};
}

