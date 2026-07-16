#!/usr/bin/env python3
"""Unify blog post filenames to YYYY-MM-DD-slug.md format."""

import os
import re
import yaml

BLOG_DIR = os.path.expanduser("~/notes/blog")

def extract_frontmatter(filepath):
    """Extract date and title from frontmatter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    m = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not m:
        return None, None
    
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, None
    
    if not fm:
        return None, None
    
    date = str(fm.get('date', '')).strip()
    title = str(fm.get('title', '')).strip()
    return date, title

def slugify(text):
    """Generate a URL-friendly slug from text."""
    # Lowercase
    text = text.lower()
    # Replace dots in version numbers with hyphens: DETR3D -> detr3d
    # Keep hyphens, alphanumeric
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    text = text.strip('-')
    return text

def main():
    renames = []
    
    for root, dirs, files in os.walk(BLOG_DIR):
        for fname in sorted(files):
            if fname == 'index.md':
                continue
            if not fname.endswith('.md'):
                continue
            
            filepath = os.path.join(root, fname)
            
            # Check if already in YYYY-MM-DD-slug.md format
            if re.match(r'^\d{4}-\d{2}-\d{2}-.+\.md$', fname):
                continue
            
            date, title = extract_frontmatter(filepath)
            if not date:
                print(f"SKIP (no date): {filepath}")
                continue
            
            # Normalize date to YYYY-MM-DD
            date_match = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date)
            if not date_match:
                print(f"SKIP (bad date format '{date}'): {filepath}")
                continue
            
            year, month, day = date_match.groups()
            date_norm = f"{year}-{int(month):02d}-{int(day):02d}"
            
            # Generate slug from existing filename (without extension)
            base = os.path.splitext(fname)[0]
            slug = slugify(base)
            
            new_fname = f"{date_norm}-{slug}.md"
            new_filepath = os.path.join(root, new_fname)
            
            if filepath == new_filepath:
                continue
            
            renames.append((filepath, new_filepath, fname, new_fname))
            print(f"RENAME: {fname} -> {new_fname}")
    
    # Perform renames
    for old, new, old_fname, new_fname in renames:
        os.rename(old, new)
    
    print(f"\nTotal: {len(renames)} files renamed.")
    
    # Update cross-references
    # Build old -> new filename mapping for relative links
    fname_map = {old_fname: new_fname for _, _, old_fname, new_fname in renames}
    
    for root, dirs, files in os.walk(BLOG_DIR):
        for fname in files:
            if not fname.endswith('.md'):
                continue
            filepath = os.path.join(root, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            for old_name, new_name in fname_map.items():
                # Match markdown links: [text](OldName.md)
                if f"]({old_name})" in content:
                    content = content.replace(f"]({old_name})", f"]({new_name})")
                    modified = True
                    print(f"LINK: {filepath}[...]({old_name}) -> [{new_name}]")
            
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

if __name__ == '__main__':
    main()
