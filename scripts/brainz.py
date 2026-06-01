""" Brain map visualisation script 

Render and save an interactive 3D scatter plot where every point represents a 
synapse and every point's colour corresponds to its assigned neurotransmitter
probability cluster.

"""
import numpy as np
import os
import pandas as pd
import pyvista as pv
import sys
from trame.widgets import vuetify
from scipy.spatial import ConvexHull
from datetime import datetime




################################################################################


def toggle(actor_name, actors, state, plotter):
    actors[actor_name].SetVisibility(state)
    plotter.update()
    

def get_convex_hull(neuropil_df):
    """ Return the convex hull surrounding the xyz coords of a given neuropil """
    coords = neuropil_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False) # Get coords
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
    
    # Set up coordinate groups
    grouped2 = connectome.groupby("cluster")
    gaba_cloud = grouped2.get_group("gaba")[["x", "y", "z"]].to_numpy(
        dtype=np.float32, copy=False)
    ach_cloud = grouped2.get_group("ach")[["x", "y", "z"]].to_numpy(
        dtype=np.float32, copy=False)
    other_cloud = grouped2.get_group("other")[["x", "y", "z"]].to_numpy(
        dtype=np.float32, copy=False)
    low_gaba_noise_cloud = grouped2.get_group("low_gaba_noise")[["x", "y", "z"]].to_numpy(
        dtype=np.float32, copy=False)
    low_ach_noise_cloud = grouped2.get_group("low_ach_noise")[["x", "y", "z"]].to_numpy(
        dtype=np.float32, copy=False)
    misc_noise_cloud = grouped2.get_group("misc_noise")[["x", "y", "z"]].to_numpy(
        dtype=np.float32, copy=False)
    
    # Add coordinates to plotter and store result for toggling vis later
    actors = {
        "gaba_actor": plotter.add_points(gaba_cloud, color="blue", point_size=3),
        "ach_actor": plotter.add_points(ach_cloud, color="yellow", point_size=3),
        "other_actor": plotter.add_points(other_cloud, color="pink", point_size=3),
        "low_gaba_noise_actor": plotter.add_points(low_gaba_noise_cloud, 
                                                   color="grey", point_size=3),
        "low_ach_noise_actor": plotter.add_points(other_cloud, color="grey", 
                                                  point_size=3),        
        "misc_noise_actor": plotter.add_points(other_cloud, color="grey", 
                                               point_size=3)
    }
    del gaba_cloud, ach_cloud, other_cloud, low_gaba_noise_cloud
    del low_ach_noise_cloud, misc_noise_cloud
    for actor in actors.values():
        actor.SetVisibility(True)

    # Save brain map
    save(plotter, actors, outdir)


def save(plotter, actors, outdir):
    """ Save 3D interactive visualisations """
    print(f"{datetime.now().strftime("%H:%M:%S")} Saving plots ...")
    
    # Export brain map with all points visible
    plotter.export_html(os.path.join(outdir, "bm_interactive.html"))
    
    # Export brain maps with only one thing toggled on
    for actor in actors.values():
        actor.SetVisibility(False)
    for actor in actors:
        actor.SetVisibility(True)
        label = actor.strip("_actor")
        plotter.export_html(os.path.join(outdir, f"bm_{label}.html"))
        actor.SetVisibility(False)
    
    # Export brain maps with only non-noise on
    actors["gaba_actor"].SetVisibility(True)
    actors["ach_actor"].SetVisibility(True)
    actors["other_actor"].SetVisibility(True)
    plotter.export_html(os.path.join(outdir, "bm_mains.html"))
    actors["gaba_actor"].SetVisibility(False)
    actors["ach_actor"].SetVisibility(False)
    actors["other_actor"].SetVisibility(False)
    
    # Export brain maps with only noise on
    actors["low_gaba_noise_actor"].SetVisibility(True)
    actors["low_ach_noise_actor"].SetVisibility(True)
    actors["misc_noise_actor"].SetVisibility(True)
    plotter.export_html(os.path.join(outdir, "bm_noise.html"))    
    
    print(f"{datetime.now().strftime("%H:%M:%S")} Plots saved")


if __name__ == "__main__":
    print("Call received!")
    inpath, outdir = sys.argv[1], sys.argv[2]
    print("Loading sampled connectome from file ...")
    print(inpath)
    connectome = pd.read_parquet(inpath)
    print("Making brain map ...")
    make_brain_map(connectome, outdir)