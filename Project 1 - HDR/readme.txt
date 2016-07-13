1. Dictionary:
	images: the original images
	program: the source code
	result: the camera response curve + the HDR file + the LDR image
2.Instruction:
	main(inputFolder,outputFolder,lambda,key,Lwhite,phi,threshold)
	ex. 
	Step1. open the "main.m" in the program folder with MATLAB
	Step2. key in the following instruction to the MATLAB Command Window
		main('AdminBuilding','AdminBuilding',22,0.36,1.5,8,0.05)
	p.s. Note that don't close the figure displaying the camera response curve until the program finished!
	When the program finishes, the command widow will show "Done!".