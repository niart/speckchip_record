import samna
import samnagui
from multiprocessing import Process
import time
from threading import Thread


class SamnaGraphBuilder:
    def __init__(self, device_name, input_device_name):
        self.device_name = device_name
        self.input_device_name = input_device_name
        self.devkit = None
        self.dk_source_node = None
        self.samna_graph = None
        self.streamer_node = None
        self.input_dev = None
        self.input_graph = None
        self.input_graph = samna.graph.EventFilterGraph()
        self.samna_graph = samna.graph.EventFilterGraph()
        self.readoutBuf = (
            samna.BasicSinkNode_dynapcnn_event_output_event()
        )  # receive the spike events

    def open_device(self):
        devices = samna.device.get_unopened_devices()
        device_names = [each.device_type_name for each in devices]
        print(device_names)
        if self.device_name in device_names:
            self.devkit = samna.device.open_device(self.device_name)
        else:
            raise ValueError("Device not found.")

    def open_input_device(self):
        if self.input_device_name != "None":
            devices = samna.device.get_unopened_devices()
            device_names = [each.device_type_name for each in devices]
            print(device_names)
            if self.input_device_name in device_names:
                self.input_dev = samna.device.open_device(self.input_device_name)
            else:
                raise ValueError("Input device not found.")

    def kill_hot_pixels(self):
        for i in range(128):
            samna.speck2e.event.KillSensorPixel(i, 128)

    def route_input(self):
        if self.device_name == "DynapcnnDevKit":
            input = "CameraToDynapcnnInput"
        elif self.device_name == "Speck2eDevKit":
            input = "CameraToSpeck2eInput"

        if self.input_device_name != "None":
            # process dvs events and put them to dynapcnn
            # Add node in filter graph and nodes are connected automatically.
            (_, dvs_crop_node, dvs_decimate_node, _, _) = self.input_graph.sequential([
                self.input_dev.get_source_node(), "DvsEventCrop", "DvsEventDecimate", input,
                self.devkit.get_model().get_sink_node()])
            # we only want the pixels inside the rectangle
            # whose top left corner is (100,100) and bottom right corner is (288,228) pass
            dvs_crop_node.set_roi(100, 100, 228, 228)
            # every 3 events we pick 1 event
            dvs_decimate_node.set_decimation_fraction(10, 1)

    def route_output(self, streamer_endpoint):
        if self.device_name == "Speck2eDevKit":
            converter = "Speck2eDvsToVizConverter"
        elif self.device_name == "DynapcnnDevKit":
            converter = "DynapcnnDvsToVizConverter"
        else:
            print("unknown device name")
        self.dk_source_node = self.devkit.get_model_source_node()
        _, _, self.streamer_node = self.samna_graph.sequential(
            [self.dk_source_node, converter, "VizEventStreamer"]
        )

        self.streamer_node.set_streamer_endpoint(streamer_endpoint)

    def add_buf_to_print(self):
        buf = samna.graph.sink_from(self.devkit.get_model_source_node())

        def record():
            while True:
                time.sleep(1)
                b = buf.get_events()
                for e in b:
                    print("output event: ", e)

        t1 = Thread(target=record)
        t1.setDaemon(True)
        t1.start()



    def start_samna_node(self):
        samna_node = samna.init_samna()
        time.sleep(1)
        return samna_node


###########define the visualizer #######################################################################################
class SamnaVisualizer:
    def __init__(self, receiver_endpoint, sender_endpoint, visualizer_id):
        self.receiver_endpoint = receiver_endpoint
        self.sender_endpoint = sender_endpoint
        self.visualizer_id = visualizer_id
        self.gui_process = None

    def run_samnagui_visualizer(self):
        width, height = 0.6, 0.6
        samnagui.runVisualizer(width, height, self.receiver_endpoint, self.sender_endpoint, self.visualizer_id)

        return

    def start(self):
        self.gui_process = Process(target=self.run_samnagui_visualizer)
        self.gui_process.start()

    def stop(self):
        if self.gui_process is not None:
            self.gui_process.terminate()
            self.gui_process.join()


####### configure the dvs layers and dvs filters #########################################################################
class SamnaConfigure:
    def __init__(self,input_device_name, device_name, devkit):
        self.input_device_name = input_device_name
        self.devkit = devkit
        self.device_name = device_name
        if self.device_name == "Speck2eDevKit":
            self.devkit_config = samna.speck2e.configuration.SpeckConfiguration()
            self.devkit_config.dvs_filter.enable = True
            self.devkit_config.dvs_filter.hot_pixel_filter_enable = True
            self.devkit_config.dvs_filter.threshold = 1
            self.devkit_config.dvs_layer.raw_monitor_enable = False
        elif self.device_name == "DynapcnnDevKit":
            self.devkit_config = samna.dynapcnn.configuration.DynapcnnConfiguration()
        #self.devkit_config.dvs_layer.pooling.x = 1
        #self.devkit_config.dvs_layer.pooling.y = 1
        # enable monitoring the inputs from the DVS sensor
        #self.devkit_config.dvs_layer.monitor_enable = True
        #self.devkit_config.dvs_layer.mirror.y = True
        #self.devkit_config.dvs_layer.mirror_diagonal = True

    def davis_and_speck(self, out_layer, network):
        self.devkit_config = network.make_config(device="speck2edevkit:0")
        new_out_channel = (out_layer - 1) * 4 + 1
        self.devkit_config.cnn_layers[new_out_channel].monitor_enable = True
        self.devkit_config.dvs_layer.destinations[0].enable = True
        self.devkit_config.dvs_layer.destinations[0].layer = network.chip_layers_ordering[0]
        if self.input_device_name == "None":
            self.devkit_config.dvs_layer.mirror_diagonal = True
        # link the dvs layer to the 1st layer of the cnn layers
        self.devkit_config.dvs_layer.merge = False
        self.devkit_config.dvs_layer.monitor_enable = False  # let it output dvs events
        self.devkit_config.dvs_layer.raw_monitor_enable = False

    def davis_and_dynapcnn(self, out_layer, network):
        chip_layers_ordering = [0, 1 ,2]
        self.devkit_config = network.make_config(chip_layers_ordering=chip_layers_ordering)
        #self.devkit_config.dvs_layer.destinations[0].enable = True
        #self.devkit_config.dvs_layer.destinations[0].layer = network.chip_layers_ordering[0]
        #self.devkit_config.dvs_layer.merge = False
        self.devkit_config.dvs_layer.monitor_enable = False  # let it output dvs events
        self.devkit_config.cnn_layers[out_layer].monitor_enable = True
        self.devkit_config.factory_settings.monitor_input_enable = False

    def apply_config(self):
        self.devkit.get_model().apply_configuration(self.devkit_config)


###### run the setup ###############################################################################################
class SamnaController:
    def __init__(self, device_name, input_device_name, visualizer_id):
        self.device_name = device_name
        self.input_device_name = input_device_name
        self.visualizer_id = visualizer_id
        self.graph_builder = SamnaGraphBuilder(device_name, input_device_name)
        self.visualizer = SamnaVisualizer(None, None, visualizer_id)
        self.streamer_endpoint = "tcp://0.0.0.0:40005"

    def setup(self):
        self.graph_builder.open_device()
        self.graph_builder.open_input_device()
        self.graph_builder.route_input()
        self.graph_builder.route_output(self.streamer_endpoint)
        self.graph_builder.add_buf_to_print()

    def start_visualizer(self):
        samna_node = self.graph_builder.start_samna_node()
        sender_endpoint = samna_node.get_sender_endpoint()
        receiver_endpoint = samna_node.get_receiver_endpoint()
        samna_graph = self.graph_builder.samna_graph
        input_graph = self.graph_builder.input_graph

        self.visualizer.receiver_endpoint = receiver_endpoint
        self.visualizer.sender_endpoint = sender_endpoint
        self.visualizer.start()

        timeout = 10
        begin = time.time()
        name = "visualizer" + str(self.visualizer_id)
        while time.time() - begin < timeout:
            try:
                time.sleep(0.05)
                samna.open_remote_node(self.visualizer_id, name)
            except:
                continue
            else:
                print("Successfully opened the visualizer")
                break

        self.visualizer = getattr(samna, name)

        #
        activity_plot_id = self.visualizer.plots.add_activity_plot(128, 128, "DVS Layer")
        plot_name = "plot_" + str(activity_plot_id)
        plot = getattr(self.visualizer, plot_name)
        # set the position: top left x, top left y, bottom right x, bottom right y
        plot.set_layout(0, 0, 0.6, 1)

        # setup the visualizer
        # set the recevier endpoint of visualizer as the streamer node's endpoint
        self.visualizer.receiver.set_receiver_endpoint(self.streamer_endpoint)
        self.visualizer.receiver.add_destination(self.visualizer.splitter.get_input_channel())
        self.visualizer.splitter.add_destination("passthrough", self.visualizer.plots.get_plot_input(activity_plot_id))

        input_graph.start()
        samna_graph.start()

    def start_camera(self, input_device_name):
        if input_device_name != "None":
            self.graph_builder.input_dev.start()

    def run(self):
        self


