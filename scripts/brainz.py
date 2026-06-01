""" Brain map visualisation script 

Render and save an interactive 3D scatter plot where every point represents a 
synapse and every point's colour corresponds to its assigned neurotransmitter
probability cluster.

"""

import numpy as np
import os
import pandas as pd
import pyvista as pv
from trame.widgets import vuetify
from scipy.spatial import ConvexHull
from datetime import datetime




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
    plotter = pv.Plotter()
    
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


if __name__ == "__main__":
    print("Call received!")
    inpath, outdir = sys.argv[2], sys.argv[3]
    print("Loading sampled connectome from file ...")
    connectome = pd.read_parquet(inpath)
    print("Making brain map ...")
    make_brain_map(connectome, outdir)