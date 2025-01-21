import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

def selectedAnnotation = QPEx.getSelectedObject()
if (!selectedAnnotation) {
    print 'No annotation selected.'
    return
}

def detections = QPEx.getCurrentHierarchy().getObjectsForROI(
    qupath.lib.objects.PathDetectionObject, selectedAnnotation.getROI()
)

if (detections.isEmpty()) {
    print 'No detections found in the selected annotation.'
    return
}

def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()

def outputDir = QPEx.buildFilePath(QPEx.PROJECT_BASE_DIR, 'export_DAPI')
QPEx.mkdirs(outputDir)

def name = server.getMetadata().getName().replaceAll(/\.[^\.]+$/, "")
def path = QPEx.buildFilePath(outputDir, name + "_DAPI-labels.tif")

def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0)
    .useInstanceLabels()
    .useFilter { obj -> detections.contains(obj) }
    .downsample(1.0)
    .multichannelOutput(false)
    .build()

QPEx.writeImage(labelServer, path)
print "Export complete: " + path
