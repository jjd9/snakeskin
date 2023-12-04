# !/usr/bin/python3

import click

from snakeskin.assign_all_mosaic import AssignAll
from snakeskin.best_mosaic import Best
from snakeskin.quadtree_mosaic import Quad
from snakeskin.superpixel_mosaic import SuperPixel

METHOD_MAP = {
    "all": AssignAll,
    "best": Best,
    "quad": Quad,
    "superpixel": SuperPixel
}

@click.command()
@click.argument('input_image_path')
@click.argument('album_path')
@click.option('--output_path', default='./', help='Directory to save the output images to (it must exist!).')
@click.option('--method', default='best', help='Mosaic method to use.')
@click.option('--scale', default=0.1, help='Scale of the mosaic tiles (relative to the input image) for processing. It has the most noticeable effect when method==best')
@click.option('--alpha', default=0.3, help='Weighted given to the input image when computing the final mosaic image.')
@click.option('--verbose', default=False, help='Whether or not to log information about the mosaic process.')
@click.option('--show_lines', default=False, help='Whether or not to draw the lines between the mosaic tiles.')
@click.option('--use_cache', default=True, help='Whether or not to use the database.pkl file in the `album` directory, if it exists.')
def hiss(input_image_path, album_path, output_path, method, scale, alpha, verbose, show_lines, use_cache):
    mosaic = METHOD_MAP[method](scale = scale, alpha = alpha, use_cache=use_cache, verbose = verbose, show_lines=show_lines)
    mosaic.create(input_image_path, album_path, output_path)

if __name__ == "__main__":
    hiss()
