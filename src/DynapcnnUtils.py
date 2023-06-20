import time
import samna
from samna.flasher import *

def open_dynapcnn(idx: int = 0):
    """Function to open the DynapCNN chip with index <idx>."""
    return samna.device.open_device("DynapcnnDevKit:" + str(idx))

def open_camera(idx: int = 0):
    """Function to open the Prophesee camera with index <idx>."""
    return samna.device.open_device("PseeCamera:" + str(idx))


def flash_chip() -> None:
    """
    Setting up the DynapCNN chip since its internal flash seems to be broken.
    """
    d = samna.flasher.get_empty_devices()
    print("Empty devices:", d)
    if len(d) > 0:
        print("DynapCNN chip is not flashed yet...")
        program_empty_fx3_ram(d[0], './deviceimages/dynapcnnDevkit.img')
        time.sleep(5)
        print(samna.device.get_all_devices())
        print("Ram was flashed!")


def initialize_IO(device, input_graph, output_graph):
    """Function to initialize the """
    # Source node and sink node
    source = samna.BasicSourceNode_dynapcnn_event_input_event()
    sink = device.get_model().get_sink_node()
    # Initialize directed input graph for event processing
    input_graph.sequential([source, sink])
    input_graph.start()

    # Source node and sink node
    buf = samna.BasicSinkNode_dynapcnn_event_output_event()
    # Initialize directed input graph for event retrieval
    output_graph.sequential([device.get_model().get_source_node(), buf])
    output_graph.start()

    return source, buf


def print_all_device_info(device_list) -> None:
    print([dev.to_json() for dev in device_list])

