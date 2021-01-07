import argparse
import os


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--output-dir', required=True, type=os.path.abspath,
    )
    argument_parser.add_argument(
        '--tmp-dir', type=os.path.abspath, default='./',
    )
    args = argument_parser.parse_args()

    # The tool that will manage the data procurement
    # https://github.com/lucasdavid/wikiart
    os.system('mkdir -p {}'.format(args.tmp_dir))
    os.system(
        'git clone https://github.com/lucasdavid/wikiart.git {}'.format(
            args.tmp_dir,
        ),
    )
    os.system(
        'python3 {}/wikiart/wikiart.py --datadir {} fetch --only artists'.format(
            args.tmp_dir, args.output_dir,
        ),
    )

    # Metadata to indicate the artist to be downloaded
    entry = """[{
        "artistName": "Claude Monet",
        "birthDay": "/Date(-4074969600000)/",
        "birthDayAsString": "November 14, 1840",
        "contentId": 211667,
        "deathDay": "/Date(-1359331200000)/",
        "deathDayAsString": "December 5, 1926",
        "dictonaries": [
            1221,
            316
        ],
        "image": "https://uploads0.wikiart.org/00115/images/claude-monet/440px-claude-monet-1899-nadar-crop.jpg!Portrait.jpg",
        "lastNameFirst": "Monet Claude",
        "url": "claude-monet",
        "wikipediaUrl": "http://en.wikipedia.org/wiki/Claude_Monet"
    }]"""

    with open(
            '{}/meta/artists.json'.format(args.output_dir), 'w',
    ) as artists_file:
        artists_file.write(entry)

    # Download the art from the artist
    os.system(
        'python3 {}/wikiart/wikiart.py --datadir {} fetch'.format(
            args.tmp_dir, args.output_dir,
        ),
    )

    os.rmdir(args.tmp_dir)
