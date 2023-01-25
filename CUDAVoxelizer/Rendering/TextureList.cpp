#include "stdafx.h"
#include "TextureList.h"

AlgGeom::TextureList::TextureList()
{
}

AlgGeom::TextureList::~TextureList()
{
    for (auto& pair : _colorTexture)
    {
        delete pair.second;
    }

    for (auto& pair : _imageTexture)
    {
        delete pair.second;
    }
}

AlgGeom::Texture* AlgGeom::TextureList::getTexture(const vec4& color)
{
    AlgGeom::Texture* texture = nullptr;
    auto it = _colorTexture.find(color);

    if (it == _colorTexture.end())
    {
        texture = new AlgGeom::Texture(color);
        _colorTexture[color] = texture;
    }
    else
        texture = it->second;

    return texture;
}

AlgGeom::Texture* AlgGeom::TextureList::getTexture(const std::string& path)
{
    AlgGeom::Texture* texture = nullptr;
    auto it = _imageTexture.find(path);

    if (it == _imageTexture.end())
    {
        try
        {
            texture = new AlgGeom::Texture(new Image(path));
            _imageTexture[path] = texture;
        }
        catch (std::runtime_error& error)
        {
            return nullptr;
        }
    }
    else
    {
        texture = it->second;
    }

    return texture;
}

void AlgGeom::TextureList::saveTexture(const vec4& color, AlgGeom::Texture* texture)
{
    _colorTexture[color] = texture;
}

void AlgGeom::TextureList::saveTexture(const std::string& path, AlgGeom::Texture* texture)
{
    _imageTexture[path] = texture;
}
