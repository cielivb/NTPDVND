""" Brain map visualisation script """
import numpy as np
import os
import pandas as pd
import pyvista as pv
from pyvista.trame.jupyter import TramePlotter
from trame.widgets import vuetify
from scipy.spatial import ConvexHull
from datetime import datetime


def _drop_unassigned(connectome, id_col):
    """ Remove edges/rows where ID is NaN """
    return connectome[~connectome[id_col].isna()].persist()


def _attach_colours(connectome, id_col):
    """ Attach column with colour group assignment to connectome.
    Edges are assigned to colour groups based on max neurotransmitter (NT)
    probability."""
    def _assign_colour_group(tagged: pd.DataFrame) -> pd.Series:
        """ Return series containing colours 'g', 'B', 'Y', 'P'
        'g' = grey, 'B' = blue, 'Y' = yellow, 'P' = pink """
        # Concatenate NT columns then get the indexes of column-wise max values
        gaba, ach, other = tagged["gaba"], tagged["ach"], tagged["other"]
        probs = pd.concat([gaba, ach, other], axis=1)
        max_prob = probs.idxmax(axis=1) # Picks arbitrarily if tied
        
        # Assign colours
        colour = pd.Series("g", index=tagged.index) # Default to grey
        is_na = tagged[id_col].isna()
        colour[(~is_na) & (max_prob == "gaba")] = "B"
        colour[(~is_na) & (max_prob == "ach")] = "Y"
        colour[(~is_na) & (max_prob == "other")] = "P"
        return colour
    
    colour_series = connectome.map_partitions(
        _assign_colour_group, meta=("colour", "object")).reset_index(drop=True)
    coloured = connectome.assign(colour=colour_series)
    return coloured.persist()


def _map_colour_codes(p):
    """ Convert letter codes to RGBA arrays """
    letter_to_rgba = {
        "B": [0, 0, 255, 255], # gaba
        "Y": [255, 255, 0, 255], # acetylcholine
        "P": [255, 0, 255, 255], # other
        "g": [128, 128, 128, 40], # unassigned
    }
    rgba = np.vstack(p["colour"].map(letter_to_rgba).to_numpy())
    p["colour"] = list(rgba)
    return p
    

def save(plotter, outdir):
    """ Save 3D interactive visualisation and above/side/front screenshots """
    print(f"{datetime.now().strftime("%H:%M:%S")} Saving plot ...")
    plotter.export_html(os.path.join(outdir, "bm_interactive.html"), backend="trame")    
    
    # Save screenshots from above, side, and front views
    plotter.reset_camera()
    plotter.view_vector((0, 0, 1)) # Above / looking down z-axis
    plotter.screenshot(os.path.join(outdir, "bm_above.png"))
    plotter.reset_camera()
    plotter.view_vector((1, 0, 0)) # Side / looking along x-axis
    plotter.screenshot(os.path.join(outdir, "bm_side.png"))
    plotter.reset_camera()
    plotter.view_vector((0, 1, 0)) # Front / looking along y-axis
    plotter.screenshot(os.path.join(outdir, "bm_front.png"))
    
    print(f"{datetime.now().strftime("%H:%M:%S")} Plot saved")
    

def get_plotter(connectome, id_col, plot_unassigned=True):
    """ Create a PyVista plotter for plotting EIOs for edges with IDs. 
    The plot shows edges with NaN IDs in either semi-transparent gray or not
    at all. Colours of classified edges is based on the highest
    neurotransmitter probability for that edge (blue = GABA, yellow =
    acetylcholien, pink = other).
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Getting PyVista plotter ...")
    data = connectome if plot_unassigned else _drop_unassigned(connectome, id_col)
    coloured = _attach_colours(data, id_col)
    
    # Can't use map_partitions because creating external effect (adding points 
    # to plotter). Convert tagged dataframe into list of delayed instead. Each 
    # delayed represents 1 partition of coloured dataframe.
    plotter = pv.Plotter(off_screen=True)
    for partition in coloured.to_delayed():
        p = partition.compute()
        p = _map_colour_codes(p)
        points = p[["x", "y", "z"]].to_numpy(dtype=np.float32)
        cloud = pv.PolyData(points)
        cloud["colour"] = np.vstack(p["colour"].to_numpy()).astype(np.uint8)
        plotter.add_points(cloud, 
                           scalars = "colour", 
                           rgba = True,
                           point_size = 2)
    print(f"{datetime.now().strftime("%H:%M:%S")} Got PyVista plotter")
    return plotter




################################################################################

def toggle(actor_name, actors, state, plotter):
    actors[actor_name].SetVisibility(state)
    plotter.update()
    

def get_convex_hull(neuropil_df):
    """ Return the convex hull surrounding the xyz coords of a given neuropil """
    coords = neuropil_df[["x", "y", "z"]].to_numpy(copy=False) # Get coords
    convex_hull_raw = ConvexHull(coords) # Get 3D convex hull
    
    # Build 1D pyvista faces array [3, i1, i2, i3, 3, i4, i5, i6, ...] from
    # 3D hull.simplices array containing [i1, i2, i3], [i4, i5, i6], etc.
    # Using 3 because 3 vertices per trianlge in the convex hull.
    faces = np.hstack([[3, *triangle] for triangle in convex_hull_raw.simplices])
    
    mesh = pv.PolyData(coords, faces) # create pyvista mesh
    return mesh


def make_brain_map(connectome, outdir):
    """ Generate an interactive brain map with neurotransmitter cluster toggles.
    Connectome should be adequately downsampled before calling this func. """
    plotter = TramePlotter()
    
    # Render neuropil outlines
    neuropils = list(connectome["neuropil"].unique())
    grouped = connectome.groupby("neuropil")
    for neuropil in neuropils:
        neuropil_df = grouped.get_group(neuropil)
        convex_hull = get_convex_hull(neuropil_df)
        del neuropil_df
        plotter.add_mesh(convex_hull, color="grey", opacity=0.05, show_edges=True)
        
    # Render connectome coordinates
    grouped2 = connectome.groupby("cluster")
    gaba_cloud = grouped2.get_group("gaba")[["x", "y", "z"]]
    ach_cloud = grouped2.get_group("ach")[["x", "y", "z"]]
    other_cloud = grouped2.get_group("other")[["x", "y", "z"]]
    none_cloud = grouped2.get_group("none")[["x", "y", "z"]]
    actors = {
        "gaba_actor": plotter.add_points(gaba_cloud, color="blue", point_size=3),
        "ach_actor": plotter.add_points(ach_cloud, color="yellow", point_size=3),
        "other_actor": plotter.add_points(other_cloud, color="pink", point_size=3),
        "none_actor": plotter.add_points(none_cloud, color="grey", point_size=3)
    }
    del gaba_cloud, ach_cloud, other_cloud, none_cloud    
    for actor in actors.values():
        actor.SetVisibility(True)

    # Set up UI with checkboxes
    with plotter.ui as ui:
        with vuetify.VContainer(fluid=True):
            for actor_name in actors:
                ui.add_checkbox(actor_name, True,
                    lambda state, name=actor_name: toggle(
                        name, actors, state, plotter))

    # Save brain map
    save(plotter, outdir)


if __name__ == "__main__":
    print("Call received!")
    inpath, outdir = sys.argv[2], sys.argv[3]
    print("Loading sampled connectome from file ...")
    connectome = pd.read_parquet(inpath)
    print("Making brain map ...")
    make_brain_map(connectome, outdir)