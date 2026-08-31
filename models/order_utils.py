import os

# use to reproduce the splits as in the paper 

def ordered_files(data_folder, order_file=None, ext=".pt"):
    """Return the graph filenames of `data_folder` in a deterministic order.

    order_file is None  -> alphabetical order (default for new experiments).
    order_file is a path -> the order listed in that file, used to reproduce
        splits from runs that predate this function, when the loader consumed
        raw os.listdir() order. The file must name every graph in the folder,
        exactly once, using current filenames.

    Raises rather than falling back, because a silently wrong order produces
    plausible-looking results with the wrong folds.
    """
    present = {
        f for f in os.listdir(data_folder)
        if f.endswith(ext) and not f.startswith("._")
    }

    if order_file is None:
        return sorted(present)

    with open(order_file) as fh:
        order = [l.strip() for l in fh if l.strip().endswith(ext)]

    seen = set()
    duplicates = {f for f in order if f in seen or seen.add(f)}
    if duplicates:
        raise ValueError(
            f"{order_file}: {len(duplicates)} duplicated entries, "
            f"e.g. {sorted(duplicates)[:3]}"
        )

    missing = [f for f in order if f not in present]
    extra = sorted(present - seen)
    if missing or extra:
        raise ValueError(
            f"{order_file} does not match {data_folder}: "
            f"{len(missing)} listed but absent (e.g. {missing[:3]}), "
            f"{len(extra)} present but unlisted (e.g. {extra[:3]})"
        )

    return order