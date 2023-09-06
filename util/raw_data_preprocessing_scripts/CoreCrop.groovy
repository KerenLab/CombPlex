import javax.imageio.ImageIO
import qupath.lib.regions.RegionRequest

// Define resolution - 1.0 means full size
double downsample = 1.0

// Create output directory inside the project
def dirOutput = buildFilePath(PROJECT_BASE_DIR, 'cores')
mkdirs(dirOutput)

def baseImageName = getProjectEntry().getImageName()

// Write the cores
def server = getCurrentImageData().getServer()
def path = server.getPath()
cores = getTMACoreList().findAll{core -> !core.isMissing()}

//for (core in cores) {
//    coreName = core.getName()
//    currentImagePath = buildFilePath(dirOutput,coreName() + '.tif')
//    writeImageRegion(server,core,currentImagePath)
//}
//def cores = getTMACoreList().findAll {core -> !core.isMissing()}
//for core in cores {
//    currentImagePath = buildFilePath(dirOutput,core.getName() + '.tif')
//    writeImageRegion(server,Core, currentImagePath)
//}

getTMACoreList().parallelStream().forEach({core ->
    def Core = RegionRequest.createInstance(path,downsample, core.getROI())
    currentImagePath = buildFilePath(dirOutput,core.getName() + '.tif')
    writeImageRegion(server,Core, currentImagePath)
})


def crop = 'Crop'
getAnnotationObjects().eachWithIndex{it, x->
        println("Working on: "+it)
	def roi = it.getROI()
	def requestROI = RegionRequest.createInstance(server.getPath(), 1, roi)
	currentImagePath = buildFilePath(dirOutput,crop +it+ '.tif')
	writeImageRegion(server, requestROI, currentImagePath)
}

print('Done!')

