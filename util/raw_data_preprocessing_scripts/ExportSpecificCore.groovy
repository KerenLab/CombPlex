def server = getCurrentServer()
def path = server.getPath()
def dirOutput = buildFilePath(PROJECT_BASE_DIR, 'cores')
def roi = getSelectedROI()
def requestROI = RegionRequest.createInstance(path, 1, roi)
currentImagePath = buildFilePath(dirOutput, 'A-1.tif')
writeImageRegion(server, requestROI, currentImagePath)


