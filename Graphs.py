import holoviews as hv
import pickle


def fixup(hist):
    return [(i, max(h, 0)) for i, h in enumerate(hist)]


def meta2graph(meta):
    curve_dict = {i: hv.Curve(fixup(hist), "generation", "score")
                  for i, hist in enumerate(meta)}
    return hv.NdOverlay(curve_dict, kdims="run")


def get_graphs():
    hv.extension('matplotlib')
    with open("genetics.pkl", 'rb') as fi:
        ch, complete_history = pickle.load(fi)
    meta_dict = {i: meta2graph(met) for i, met in complete_history.items()}
    big_graph = hv.NdLayout(meta_dict,
                            kdims=["Pop. size", "Pbreed", "Pmutate"])
    big_graph.opts(hv.opts.NdOverlay(show_legend=False))
    hv.save(big_graph, 'big_graph.svg')


if __name__ == '__main__':
    get_graphs()
