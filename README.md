# Algoritmos Geométricos (2022-2023)

![c++](https://img.shields.io/github/languages/top/AlfonsoLRz/AG2223) 
![opengl](https://img.shields.io/badge/opengl-4.5-red.svg) 
![imgui](https://img.shields.io/badge/imgui-1.82-green.svg) 
![license](https://img.shields.io/badge/license-MIT-blue.svg)

Proyecto base para la asignatura de Algoritmos Geométricos en la Universidad de Jaén (Curso 2023). A continuación se describen las interacciones disponibles en la aplicación.

1. Cámara: interacción a través de teclado y ratón.
    | Movimiento | Interacción |
    | ----------- | ----------- |
    | Forward | Botón derecho + <kbd>W</kbd> |
    | Backwards | Botón derecho + <kbd>S</kbd> |
    | Left | Botón derecho + <kbd>A</kbd> |
    | Backwards | Botón derecho + <kbd>D</kbd> |
    | Zoom | Rueda de ratón |
    | Órbita horizontal | <kbd>X</kbd> |
    | Órbita vertical | <kbd>Y</kbd> |
    | Giro de cámara | Botón izquierdo de ratón |
    | Reiniciar cámara | <kbd>B</kbd> |

2. Guizmos: interacción con modelos para llevar a cabo operaciones de traslación, rotación y escalado. Para ello es necesario abrir el menú `Settings` > `Models` y seleccionar un modelo.
    | Operación | Interacción |
    | ----------- | ----------- |
    | Traslación de modelo | <kbd>T</kbd> |
    | Rotación de modelo | <kbd>R</kbd> |
    | Escalado de modelo | <kbd>S</kbd> |

<p align="center">
    <img src="readme_assets/guizmo.png" width=800 /></br>
    <em>Transformación de modelo mediante la interfaz. En este caso se muestra una rotación.</em>
</p>

3. Visualización de diferentes topologías, habiéndose generado estas durante la carga de los modelos y encontrándose en el vector de modelos de la escena (`SceneContent`, aunque debe gestionarse desde el `Renderer`). La topología se puede controlar a nivel global, de manera que podemos activar y desactivas su renderizado en el menú `Settings > Rendering`, o a nivel local (para cada modelo) mediante el menú `Settings > Models`.
    | Operación | Interacción |
    | ----------- | ----------- |
    | Activar/Desactivar nube de puntos | <kbd>0</kbd> |
    | Activar/Desactivar malla de alambre | <kbd>1</kbd> |
    | Activar/Desactivar malla de triángulos | <kbd>2</kbd> |

<p align="center">
    <img src="readme_assets/topology.png" width=500 /></br>
    <em>Nube de puntos, malla de alambre y malla de triángulos visualizadas sobre el mismo modelo.</em>
</p>

4. Captura de pantalla con antialiasing (para la documentación `:D`). Podemos realizar la captura mediante teclado o interfaz (menú `Settings > Screenshot`). Con esta última opción también es posible modificar el tamaño de la imagen o el fichero destino.
    | Operación | Interacción |
    | ----------- | ----------- |
    | Captura de pantalla | <kbd>K</kbd> |

Desde la interfaz se ofrecen otras tantas funcionalidades:

1. `Settings` > `Rendering`: 
    - Modificación de topologías visibles, como en el punto 3 de la lista previa. 
    - Modificación de color de fondo.
2. `Settings` > `Camera`:
    - Modificación de propiedades de la cámara y tipo de proyección.
3. `Settings` > `Lights`: 
    - Modificación de una única luz puntual (colores y posición en el espacio). Recordad que el objetivo de esta asignatura no es el _rendering_, por lo que nos basta con esta luz puntual que nos permitirá ver cualquier malla de triángulos situada en un punto cualquiera del espacio.

<p align="center">
    <img src="readme_assets/light_configuration.png" width=600 /></br>
    <em>Menú de configuración de la luz puntual de la escena.</em>
</p>

4. `Settings` > `Models`:
    - Modificación de transformación de modelo.
    - Modificación de material (puntos, líneas y triángulos).
    - Modificación de tamaño y anchura de puntos y líneas, respectivamente. 
    - Carga de mallas de triángulos (`.obj`, `.gltf` y `.fbx`).

## Integración de nuevos modelos renderizables

El nuevo modelo deberá implementarse como una subclase de `Model3D`, la cual nos dará todas las funcionalidades necesarias para cargar y dibujar la geometría y topología en GPU. Por tanto, nos desentendemos de esta tarea y nuestra única labor es definir geometría y topología. 

Se debe tener en cuenta que los atributos de un vértice (`VAO::Vertex`) son (por orden): posición (`vec3`), normal (`vec3`) y coordenadas de textura (`vec2`).
Así, podemos añadir nuevos vértices a nuestro modelo mediante la siguiente sintaxis:

    componente->_vertices.insert(component->vertices.end(), { vertices })

donde vertices puede definirse como sigue:

    {   
        VAO::Vertex { vec3(x, y, z), vec3(nx, ny, nz) },
        VAO::Vertex { vec3(x, y, z) },
        VAO::Vertex { vec3(x, y, z), vec3(nx, ny, nz), vec2(u, v) }
    }

El orden es importante, pero podemos omitir aquellos atributos que desconocemos.

Respecto a la topología, tendremos disponibles tres vectores (nube de puntos, malla de alambre, y malla de triángulos) en la variable `component->_indices`. De nuevo, podemos insertar primitivas como se muestra a continuación:

- Triángulos: 
    
        componente->_indices[VAO::IBO_TRIANGLES].insert(
            componente->_indices[VAO::IBO_TRIANGLES].end(), 
            { 
                0, 1, 2, RESTART_PRIMITIVE_INDEX,
                1, 2, 3, RESTART_PRIMITIVE_INDEX,
                ...
            })

- Líneas: 
    
        componente->_indices[VAO::IBO_TRIANGLES].insert(
            componente->_indices[VAO::IBO_TRIANGLES].end(), 
            { 
                0, 1, RESTART_PRIMITIVE_INDEX,
                1, 2, RESTART_PRIMITIVE_INDEX,
                ...
            })

- Puntos: 
    
        componente->_indices[VAO::IBO_TRIANGLES].insert(
            componente->_indices[VAO::IBO_TRIANGLES].end(), 
            { 
                0, 1, 2, 3, 4
                ...
            })
    
    **Nota**: dado un número de vértices `n`, podemos generar un vector como { 0, 1, 2, ..., n-1 } mediante `std::iota(begin, end, 0)` tras `vector.resize(n)`.

Además, en el menú `Settings > Models` se mostrará una lista de objetos disponibles en la escena. Por limitaciones de C++ en la herencia no es posible obtener el nombre de la clase a la que pertenece un objeto que hereda de `Model3D` en su constructor. No obstante, una vez construido es posible acceder a dicho nombre. Por tal razón, si deseamos que los objetos tengan un nombre significativo podemos hacer uso de la función `overrideModelName`.

Los métodos `SET` de la clase `Model3D` se han implementado de tal manera que puedan encadenarse las llamadas en una misma línea tras construir el objeto, incluyendo operaciones como `overrideModelName`, `setPointColor`, `setLineColor` o `setTopologyVisibility`.

## Gestión de escena

La gestión de los elementos de la escena se llevará a cabo en `Graphics/Renderer`. Para ello, disponemos de dos métodos básicos: `createModels` y `createCamera`, donde ambos generarán modelos y cámaras que se almacenaran en una instancia de `SceneContent`. Por tanto, utilizad simplemente las funciones `create*` en el `Renderer`. 

Teniendo en cuenta que la cámara puede situarse en función de los modelos, primero crearemos éstos últimos. En `buildFooScene()` tenemos un ejemplo de generación de la escena:

    vec2 minBoundaries = vec2(-1.5, -.5), maxBoundaries = vec2(-minBoundaries);

    // Triangle mesh
    auto model = (new DrawMesh())->loadModelOBJ("Assets/Models/Ajax.obj");
    model->moveGeometryToOrigin(model->getModelMatrix(), 10.0f);
    _content->addNewModel(model);


    // Spheric randomized point cloud
    int numPoints = 800, numPointClouds = 6;
    
    for (int pcIdx = 0; pcIdx < numPointClouds; ++pcIdx)
    {
        PointCloud* pointCloud = new PointCloud;

        for (int idx = 0; idx < numPoints; ++idx)
        {
            ...
            pointCloud->addPoint(Point(rand.x, rand.y));
        }

        _content->addNewModel((new DrawPointCloud(*pointCloud))->setPointColor(RandomUtilities::getUniformRandomColor())->overrideModelName());
        delete pointCloud;
    }

    // Random segments
    int numSegments = 8;

    for (int segmentIdx = 0; segmentIdx < numSegments; ++segmentIdx)
    {
        ...
        SegmentLine* segment = new SegmentLine(a, b);

        _content->addNewModel((new DrawSegment(*segment))->setLineColor(RandomUtilities::getUniformRandomColor())->overrideModelName());
        delete segment;
    }

    // Random triangles
    int numTriangles = 30;
    float alpha = ...;

    for (int triangleIdx = 0; triangleIdx < numTriangles; ++triangleIdx)
    {
        ...
        Triangle* triangle = new Triangle(a, b, c);

        _content->addNewModel((new DrawTriangle(*triangle))->setLineColor(RandomUtilities::getUniformRandomColor())->setTriangleColor(vec4(RandomUtilities::getUniformRandomColor(), 1.0f))
            ->overrideModelName());
        delete triangle;
    }

A tener en cuenta:

- `addNewModel` recibirá un puntero de un objeto que herede de `Model3D`.
- `_content` será la escena (no debemos modificar nada en esta clase).
- Los `setters` de un modelo 3D se han implementado como `Model3D* set*()` para poder encadenar llamadas en la misma instanciación (considerando que no será necesaria dicha instancia en nuestro `Renderer`, y por tanto, se eliminará a continuación).
    - ¿Qué podemos modificar mediante `setters`?:
        - Color: `setPointColor`, `setLineColor`, `setTriangleColor`. Ten en cuenta que esta última recibe un `vec4` para poder modificar el alpha del modelo.
        - Visibilidad de primitivas: `setTopologyVisibility`. Recibirá un tipo de primitiva de `VAO::IBO_slots` y un booleano.
        - `moveGeometryToOrigin`: calcula la matriz de transformación que lleva un modelo, ubicado en un punto desconocido, al origen del sistema de coordenadas. Además, se puede controlar la escala para que pueda visualizarse en nuestro viewport.
        - `overrideModelName`: por defecto, un modelo recibirá en su contructor un nombre genérico, como `Model3D 8, Comp. 0`. No obstante, podemos personalizar este nombre automáticamente para que sea identificable en la lista de modelos (accesible mediante el menú `Settings > Models`). Ten en cuenta que el nombre de un subclase no puede obtenerse en el constructor. Por tanto, se ofrece esta posibilidad como una llamada posterior.

<table style="margin:auto; width:80%">
<tr>
    <td>
        <img align="center" src="readme_assets/generic_names.png"/>
    </td>
    <td>
        <img src="readme_assets/customized_names.png"/>
    </td>
</tr>
</table>
<em>Comparativa de listado de modelos, utilizando nombres genéricos y nombres personalizados para cada modelo (asignados automáticamente).</em>

