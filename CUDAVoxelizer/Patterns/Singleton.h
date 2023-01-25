#pragma once

#include "stdafx.h"

template<typename T>
class Singleton
{
protected:
	static std::unique_ptr<T> _instance;		

protected:
	Singleton() {};
	~Singleton() {};

public:
	Singleton(const Singleton&) = delete;
	Singleton& operator=(const Singleton) = delete;
	static T* getInstance();
};

// Static members initialization

template<typename T>
std::unique_ptr<T> Singleton<T>::_instance;

// Public methods

template<typename T>
T* Singleton<T>::getInstance()
{
	if (!_instance.get())
	{
		_instance = std::unique_ptr<T>(new T());
	}

	return _instance.get();
}

