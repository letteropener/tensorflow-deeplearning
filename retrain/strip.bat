python C:\Users\danie\PycharmProjects\tensorflow\tensorflow-master\tensorflow\python\tools\strip_unused.py ^
--input_graph output_graph.pb ^
--output_graph stripped_graph.pb ^
--input_node_names "Mul" ^
--output_node_names "final_result" ^
--input_binary true
pause