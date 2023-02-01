#pragma once

#include "stdafx.h"

namespace FileStorageUtilities
{
	/**
	*	@brief Writes an string.
	*/
	template<typename T>
	void writeString(const std::string& filename, const std::vector<T>& objects);
}

template<typename T>
inline void FileStorageUtilities::writeString(const std::string& filename, const std::vector<T>& objects)
{
	std::ofstream file(filename);
	if (!file.is_open()) throw std::runtime_error("An error occurred when writing in " + filename);

	for (const T& object : objects)
	{
		file << object;
		file << '\n';
	}

	file.close();
}
