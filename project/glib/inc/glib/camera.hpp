// Made by Matthew Strumillo 2025.03.20
//
#pragma once
#include "gl.hpp"

namespace CAMERA {

	struct Camera3D {

		enum Type : u8 {
			TYPE_FREE	= 0,
			TYPE_TARGET	= 1,
		};

		u8 type;
		glm::vec3 position;
		float fov;

		union {
			glm::vec3 rotation;
			glm::vec3 target;
		};

	};

	struct Camera2D {

		glm::vec3 position;

	};

	void GetView (glm::mat4& view, const Camera3D& camera) {

		// TODO: It might be beneficial to prep. direction, right, up memory before calculation here.
		// making a reserved variable inside Camera3D Object.

		if (camera.type == Camera3D::TYPE_TARGET) {

			const glm::vec3 baseUp 	= glm::vec3 (0.0f, 1.0f,  0.0f); 
			glm::vec3 direction 	= glm::normalize (camera.position - camera.target);
			glm::vec3 right			= glm::normalize (glm::cross (baseUp, direction));
			glm::vec3 up			= glm::cross (direction, right);

			view = glm::lookAt (
				camera.position, 
				camera.target, 
				up
			);

		} else {

			const float rotX = -90.0f + camera.rotation.x;
			const float rotZ = 90.0f + camera.rotation.z;

			glm::vec3 direction;
			glm::vec3 up;

			direction.x = cos (glm::radians (rotX)) * cos (glm::radians (camera.rotation.y));
			direction.y = sin (glm::radians (camera.rotation.y));
			direction.z = sin (glm::radians (rotX)) * cos (glm::radians (camera.rotation.y));
		
			// UP is roll(z)
			up.x = cos (glm::radians (rotZ));
			up.y = sin (glm::radians (rotZ));
			up.z = 0.0;
		
			view = glm::lookAt (
				camera.position, 
  				camera.position + glm::normalize (direction), 
  				up
			);

		}

	}

}
