import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

// Obtener la anotación seleccionada
def selectedAnnotation = QPEx.getSelectedObject()
if (!selectedAnnotation) {
    print 'No annotation selected! Selecciona una anotación primero.'
    return
}

// Obtener todos los objetos de detección (células) dentro de la anotación seleccionada
def detections = QPEx.getCurrentHierarchy().getObjectsForROI(
    qupath.lib.objects.PathDetectionObject, selectedAnnotation.getROI()
)

if (detections.isEmpty()) {
    print 'No detections found in the selected annotation.'
    return
}

// Obtener datos de la imagen actual
def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()

// Configurar la carpeta de salida
def outputDir = QPEx.buildFilePath(QPEx.PROJECT_BASE_DIR, 'export_DAPI')
QPEx.mkdirs(outputDir)

// Nombre del archivo exportado
def name = server.getMetadata().getName().replaceAll(/\.[^\.]+$/, "") // Eliminar extensión
def path = QPEx.buildFilePath(outputDir, name + "_DAPI-labels.tif")

// Crear un filtro para incluir solo las detecciones seleccionadas (células)
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0) // Fondo con valor 0
    .useInstanceLabels() // Etiquetas únicas para cada célula (reemplazo de useUniqueLabels)
    .useFilter { obj -> detections.contains(obj) } // Filtrar solo las detecciones de la anotación seleccionada
    .downsample(1.0)    // Resolución completa
    .multichannelOutput(false) // Imagen etiquetada en lugar de multicanal
    .build()

// Exportar la máscara etiquetada
QPEx.writeImage(labelServer, path)

print "Exportación completada: " + path
