import os


def slugify(s):
    return s.replace('-', '_').replace(' ', '_').replace('/', '_').replace('.', '_').replace('.json', '')

def get_output_file_path(args, output_file_name):

    os.makedirs(os.path.join(slugify(args.coding_prompt), slugify(args.dataset), slugify(args.model)), exist_ok=True)
    output_file_path = os.path.join(slugify(args.coding_prompt), slugify(args.dataset), slugify(args.model), slugify(output_file_name))
    return output_file_path

def get_input_file_path(args):
    return os.path.join('datasets', slugify(args.dataset))