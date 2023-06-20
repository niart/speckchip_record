import src.samnagraph as samnagraph
import src.sinabsnetwork as sinabsnetwork
import src.DynapcnnUtils as utils
import time
# flash the dynapcnn chip if necessary
utils.flash_chip()
# Define the device name and visualizer ID
if __name__ == '__main__':
    device_name = "Speck2eDevKit" # "Speck2eDevKit" or "DynapcnnDevKit"
    input_device_name = "None"  # "Davis346" or "None"
    weights = "ones" # "ones" or "gabor"
    visualizer_id = 3

    # define neural network
    snn = sinabsnetwork.SinabsNeuralNetwork()
    network = snn.two_layer_network(input_device_name=input_device_name, weight_matrix=weights, kernel=2) # kernel has to be 10 for gabor

    # Create an instance of SamnaController
    controller = samnagraph.SamnaController(device_name, input_device_name, visualizer_id)
    # Set up the controller
    controller.setup()
    controller.start_visualizer()

    # configure the visualization
    configure = samnagraph.SamnaConfigure(input_device_name, device_name, controller.graph_builder.devkit)
    # configure the output of the network
    if device_name == "Speck2eDevKit":
        configure.davis_and_speck(out_layer=1, network=network)
    elif device_name == "DynapcnnDevKit":
        configure.davis_and_dynapcnn(out_layer=1, network=network)

    # Start the camera
    controller.start_camera(input_device_name)
    # Run the controller
    controller.run()






