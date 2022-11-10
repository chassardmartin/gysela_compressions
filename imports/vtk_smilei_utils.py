class VTKfile:
    def __init__(self):
        try:
            import vtk
        except Exception as e:
            print("Python module 'vtk' not found. Could not export to VTK format")
            return

        self.vtk = vtk

    def Array(self, data, name):
        """
		Convert a numpy array `data` in a vtkFloatArray
		"""

        from numpy import float32, int32

        shape = data.shape
        if len(shape) == 1:
            npoints, nComponents = shape[0], 1
        elif len(shape) == 2:
            npoints, nComponents = shape
        else:
            raise Exception("In Array: bad shape " + str(shape))

        if data.dtype == int32:
            arr = self.vtk.vtkIntArray()
        elif data.dtype == float32:
            arr = self.vtk.vtkFloatArray()
        else:
            raise Exception(
                "In Array: Unknown data type for data (" + str(data.dtype) + ")"
            )

        arr.SetNumberOfTuples(npoints)
        arr.SetNumberOfComponents(nComponents)
        # Replace the pointer in arr by the pointer to the data
        arr.SetVoidArray(data, npoints * nComponents, 1)
        arr.SetName(name)
        # keep reference to "data"
        # vtk.1045678.n5.nabble.com/More-zero-copy-array-support-for-Python-td5743662.html
        arr.array = data
        return arr

    def WriteImage(self, array, origin, extent, spacings, file, numberOfPieces):
        """
		Create a vtk file that describes an image
		"""
        img = self.vtk.vtkImageData()
        img.SetOrigin(origin)
        img.SetExtent(extent)
        img.SetSpacing(spacings)
        img.GetPointData().SetScalars(array)
        writer = self.vtk.vtkXMLPImageDataWriter()
        writer.SetFileName(file)
        writer.SetNumberOfPieces(numberOfPieces)
        writer.SetEndPiece(numberOfPieces - 1)
        writer.SetStartPiece(0)
        if float(self.vtk.VTK_VERSION[:3]) < 6:
            writer.SetInput(img)
        else:
            writer.SetInputData(img)
        writer.Write()

    def WriteRectilinearGrid(self, dimensions, xcoords, ycoords, zcoords, array, file):
        """
		Create a vtk file that describes gridded data
		"""
        grid = self.vtk.vtkRectilinearGrid()
        grid.SetDimensions(dimensions)
        grid.SetXCoordinates(xcoords)
        grid.SetYCoordinates(ycoords)
        grid.SetZCoordinates(zcoords)
        grid.GetPointData().SetScalars(array)
        writer = self.vtk.vtkRectilinearGridWriter()
        writer.SetFileName(file)
        if float(self.vtk.VTK_VERSION[:3]) < 6:
            writer.SetInput(grid)
        else:
            writer.SetInputData(grid)
        writer.Write()

    def WriteCloud(self, pcoords, attributes, data_format, file):
        """
		Create a vtk file that describes a cloud of points (using vtkPolyData)
		
		* pcoords: vtk array that describes the point coordinates
		* attributes: vtk arrays containing additional values for each point
		* data_format: the output data format
		* file: output file path
		"""

        points = self.vtk.vtkPoints()
        points.SetData(pcoords)

        pdata = self.vtk.vtkPolyData()
        pdata.SetPoints(points)

        # Add scalars for xml
        if data_format == "xml":

            for attribute in attributes:
                # AddArray creates scalar and then fields
                pdata.GetPointData().AddArray(attribute)

            # The first attribute (first scalar) is the main one
            if len(attributes) > 0:
                pdata.GetPointData().SetActiveScalars(attributes[0].GetName())

            writer = self.vtk.vtkXMLDataSetWriter()

        # Add scalars for vtk
        else:

            if len(attributes) > 0:
                pdata.GetPointData().SetScalars(attributes[0])
                pdata.GetPointData().SetActiveScalars(attributes[0].GetName())

            writer = self.vtk.vtkPolyDataWriter()

        writer.SetFileName(file)
        if float(self.vtk.VTK_VERSION[:3]) < 6:
            writer.SetInput(pdata)
        else:
            writer.SetInputData(pdata)
        writer.Write()

        # Add the following attributes by hand because the API limits vtk to 1 scalar
        if data_format == "vtk":
            file_object = open(file, "a")
            for attribute in attributes[1:]:
                if attribute.GetDataType() == 6:
                    data_type = "int"
                elif attribute.GetDataType() == 10:
                    data_type = "float"
                size = attribute.GetSize()
                file_object.write("POINT_DATA {} \n".format(pdata.GetNumberOfPoints()))
                file_object.write(
                    "SCALARS {} {} \n".format(attribute.GetName(), data_type)
                )
                file_object.write("LOOKUP_TABLE default \n")
                for i in range(0, size, 8):
                    remaining = min(size - i, 8)
                    for j in range(remaining):
                        file_object.write("{} ".format(attribute.GetValue(i + j)))
                    file_object.write("\n")

    def WritePoints(self, pcoords, file):
        """
		Create a vtk file that describes a set of points
		"""
        points = self.vtk.vtkPoints()
        points.SetData(pcoords)
        grid = self.vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        writer = self.vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(file)
        if float(self.vtk.VTK_VERSION[:3]) < 6:
            writer.SetInput(grid)
        else:
            writer.SetInputData(grid)
        writer.Write()

    def WriteLines(self, pcoords, connectivity, attributes, data_format, file):
        """
		Create a vtk file that describes lines such as trajectories
		
		* pcoords: vtk array that describes the point coordinates
		* connectivity: connection betwwen coordinates in pcoords to form trajectories
		* attributes: vtk arrays containing additional values for each point
		* data_format: the output data format
		* file: output file path
		"""
        ncel = len(connectivity)
        connectivity = connectivity.flatten()

        id = self.vtk.vtkIdTypeArray()
        id.SetNumberOfTuples(connectivity.size)
        id.SetNumberOfComponents(1)
        id.SetVoidArray(connectivity, connectivity.size, 1)
        connec = self.vtk.vtkCellArray()
        connec.SetCells(ncel, id)

        points = self.vtk.vtkPoints()
        points.SetData(pcoords)

        pdata = self.vtk.vtkPolyData()
        pdata.SetPoints(points)
        pdata.SetLines(connec)

        # Add scalars
        for attribute in attributes:
            pdata.GetPointData().AddArray(attribute)

        # The first attribute (first scalar) is the main one
        if len(attributes) > 0:
            pdata.GetPointData().SetActiveScalars(attributes[0].GetName())

        writer = self.vtk.vtkPolyDataWriter()

        # For XML output
        if data_format == "xml":
            writer = self.vtk.vtkXMLDataSetWriter()

        writer.SetFileName(file)
        if float(self.vtk.VTK_VERSION[:3]) < 6:
            writer.SetInput(pdata)
        else:
            writer.SetInputData(pdata)
        writer.Write()
