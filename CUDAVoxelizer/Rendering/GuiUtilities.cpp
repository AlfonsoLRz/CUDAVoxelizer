#include "stdafx.h"
#include "GuiUtilities.h"

void AlgGeom::GuiUtilities::leaveSpace(unsigned numSlots)
{
	for (unsigned i = 0; i < numSlots; ++i) ImGui::Spacing();
}

void AlgGeom::GuiUtilities::renderText(const vec3& xyz, const std::string& title, char delimiter)
{
	std::string txt = title + (title.empty() ? "" : ": ") + std::to_string(xyz.x) + delimiter + ' ' + std::to_string(xyz.y) + delimiter + ' ' + std::to_string(xyz.z);
	ImGui::Text(txt.c_str());
}
