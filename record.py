import src.samnagraph as samnagraph
import src.sinabsnetwork as sinabsnetwork
import src.DynapcnnUtils as utils
import time
import csv
# flash the dynapcnn chip if necessary
utils.flash_chip()
# Define the device name and visualizer ID


# Record events to a CSV file
def stop_recording():
    global stop_recording_flag
    stop_recording_flag = True

def record_events_to_csv():
    global stop_recording_flag
    filename = "record.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "x", "y", "p"])
    stop_recording_flag = False
    while not stop_recording_flag:
        time.sleep(1)

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

    # ... (rest of the main program remains the same)

    # Record events to a CSV file
    stop_recording_flag = False
    #stop_recording_thread = Thread(target=record_events_to_csv)
    #stop_recording_thread.setDaemon(True)
    #stop_recording_thread.start()

    try:
        # Run the controller until you want to stop recording
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop recording and close the CSV file when you interrupt the program
        stop_recording()
