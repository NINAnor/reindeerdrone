import marimo

__generated_with = "0.9.4"
app = marimo.App(width="medium", app_title="Inspect bounding boxes")


@app.cell
def __():
    import marimo as mo
    import json
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import seaborn as sns
    from collections import defaultdict
    import pandas as pd
    import altair as alt
    return alt, cv2, defaultdict, json, mo, np, patches, pd, plt, sns


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Inspecting bounding boxes""")
    return


@app.cell
def __(json):
    annotation_path = "/home/taheera.ahmed/data/reindeerdrone/tiles/train/new_annotations.json"
    img_path = "/home/taheera.ahmed/data/reindeerdrone/tiles/train/images"

    with open(annotation_path, 'r') as file:
        data = json.load(file)
    annotations = data

    images = data['images']
    annotations = data['annotations']
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    unique_annotated_images = set()

    for annotation in annotations:
        # Add the image_id from each annotation to the set, which ensures uniqueness
        unique_annotated_images.add(annotation['image_id'])

    # Count of unique images that have annotations
    number_of_annotated_images = len(unique_annotated_images)
    print(f"Number of annotated tiles: {number_of_annotated_images}")
    return (
        annotation,
        annotation_path,
        annotations,
        categories,
        data,
        file,
        images,
        img_path,
        number_of_annotated_images,
        unique_annotated_images,
    )


@app.cell
def __(annotations, categories, cv2, images, img_path, patches, plt):
    def plot_image_annotations(img_path, image_info, annotations, categories):
        # Load the image
        image_path = f"{img_path}/{image_info['file_name']}"  # Adjust the path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        
        # Initialize bounding box counter
        bbox_count = 0
        
        # Find and plot annotations for this image
        for anno in annotations:
            if anno['image_id'] == image_info['file_name']:
                x, y, width, height = anno['bbox']
                # Clip the bounding box to the image boundaries
                x_end = min(x + width, image.shape[1])
                y_end = min(y + height, image.shape[0])
                width = x_end - x
                height = y_end - y

                # Ensure bounding box is within the image
                if x >= 0 and y >= 0 and x_end <= image.shape[1] and y_end <= image.shape[0]:
                    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    
                    # Increment the bounding box counter
                    bbox_count += 1
                    
                    # Optional: Add text label for each bbox
                    label = categories.get(anno['category_id'], 'Unknown')
                    # Place label above the bounding box, outside it
                    label_x = x
                    label_y = y - 3  # Slight vertical offset from the top of the bbox
                    if label_y < 0:  # If the label goes above the image, adjust it to be just at the top inside the bbox
                        label_y = y + 3
                    
                    ax.text(label_x, label_y, label, color='white', fontsize=10, verticalalignment='bottom',
                            bbox=dict(facecolor='red', edgecolor='none', boxstyle='round,pad=0.1'))
        
        ax.set_axis_off()
        plt.show()

        # Print the number of bounding boxes
        print(f"Number of bounding boxes in image {image_info['file_name']}: {bbox_count}")


    for image_info in images[1:2]:
        plot_image_annotations(img_path, image_info, annotations, categories)

    return image_info, plot_image_annotations


@app.cell
def __(annotations, images, pd):
    df_annotations = pd.DataFrame(annotations)
    df_images = pd.DataFrame(images)
    df_annotations, df_images
    return df_annotations, df_images


@app.cell
def __(df_annotations, df_images):
    merged_df = df_annotations.merge(df_images, left_on='image_id', right_on='file_name')
    merged_df.columns
    return (merged_df,)


@app.cell
def __(alt, categories, merged_df):
    _categories = {0: 'Adult', 1: 'Calf'}

    # Calculate the count of annotations by category
    _category_counts = merged_df['category_id'].map(categories).value_counts().reset_index()
    _category_counts.columns = ['Category', 'Count']

    # Create the bar chart using Altair
    _bar_chart = alt.Chart(_category_counts).mark_bar(color='orange').encode(
        alt.X('Category:N', title='Category'),
        alt.Y('Count:Q', title='Number of Annotations'),
        tooltip=['Category:N', 'Count:Q']
    ).properties(
        title='Category Distribution',
        width=800,
        height=400
    )

    # Display the chart
    _bar_chart
    return


@app.cell
def __(alt, merged_df):
    # Count annotations per image
    _annotations_per_image = merged_df['image_id'].value_counts().reset_index()
    _annotations_per_image.columns = ['Image ID', 'Annotation Count']

    # Create a histogram using Altair
    _histogram = alt.Chart(_annotations_per_image).mark_bar(color='skyblue').encode(
        alt.X('Annotation Count:Q', bin=alt.Bin(maxbins=20), title='Number of Annotations'),
        alt.Y('count()', title='Number of Images'),
        tooltip=['Annotation Count:Q', 'count()']
    ).properties(
        title='Histogram of Annotation Counts per Image',
        width=800,
        height=400
    ).interactive()  # Enable zoom and pan

    # Display the histogram
    _histogram.show()
    return


@app.cell
def __(alt, merged_df, pd):
    # Calculate average area of bounding boxes
    total_area = merged_df['area'].sum()
    average_area = total_area / len(merged_df)

    # Create a DataFrame for the areas and categories
    df_areas = pd.DataFrame({
        'Area': merged_df['area'],
        'Category': merged_df['category_id'],
        'Image ID': merged_df['image_id']
    })

    # Define color scheme for categories
    color_scale = alt.Scale(
        domain=[0, 1],  # Assuming category 0 = Adult, 1 = Calf
        range=['#1f77b4', '#ff7f0e']
    )

    # Plot the histogram using Altair with tooltips, grouping by category
    _histogram = alt.Chart(df_areas).mark_bar().encode(
        alt.X('Area:Q', bin=alt.Bin(maxbins=100), title='Area'),
        alt.Y('count()', title='Total Frequency'),
        alt.Color('Category:N', scale=color_scale, title='Category'),  # Color by category
        tooltip=[alt.Tooltip('count()', title='Total Count'),  # Shows total count for the entire bar
                 alt.Tooltip('Area:Q', title='Area Range'),
                alt.Tooltip('Image ID', title='Image ID')]
    ).properties(
        title='Histogram of Bounding Box Areas per Category (Total Count per Bar)',
        width=600,
        height=400
    ).interactive()  # Enable zoom and pan

    # Display the chart
    _histogram.show()
    return average_area, color_scale, df_areas, total_area


@app.cell
def __(annotations, np, pd):
    _data = []
    for box in annotations:
        width = box['bbox'][2]
        height = box['bbox'][3]
        aspect_ratio = width / height
        _data.append({
            'image_id': box['image_id'],
            'category_id': box['category_id'],
            'aspect_ratio': aspect_ratio
        })

    df_aspect_ratio = pd.DataFrame(_data)

    df_aspect_ratio['log_aspect_ratio'] = np.log10(df_aspect_ratio['aspect_ratio'].replace(0, np.nan).dropna())
    df_aspect_ratio.describe()
    return aspect_ratio, box, df_aspect_ratio, height, width


@app.cell
def __(alt, df_aspect_ratio):
    histogram_log = alt.Chart(df_aspect_ratio).mark_bar().encode(
        alt.X('log_aspect_ratio:Q', bin=alt.Bin(step=0.1), title='Log10(Aspect Ratio)'),
        alt.Y('count()', title='Count'),
        color='category_id:N',
        tooltip=[alt.Tooltip('count()', title='Count'),
                 alt.Tooltip('aspect_ratio:Q', title='Original Aspect Ratio'),
                 alt.Tooltip('image_id:N', title='Image ID')]
    ).properties(
        title='Interactive Log-Transformed Aspect Ratio Distribution by Category',
        width=900,
        height=400
    ).interactive()

    # Display the interactive histogram
    histogram_log.show()
    return (histogram_log,)


@app.cell
def __(annotations, categories, images, img_path, plot_image_annotations):
    for _image_info in images:
        if _image_info['id'] == 'DSC00546_tile50':
            plot_image_annotations(img_path, _image_info, annotations, categories)
    return


@app.cell
def __(alt, merged_df):
    avg_area_per_image_per_class = merged_df.groupby(['image_id', 'category_id']).agg({'area': 'median'}).reset_index()

    # Sorting the avg_area_per_image_per_class dataframe by the 'area' column
    avg_area_per_image_per_class_sorted = avg_area_per_image_per_class.sort_values(by='area', ascending=False)

    # Create an Altair chart with sorted bars
    chart = alt.Chart(avg_area_per_image_per_class_sorted).mark_bar().encode(
        x=alt.X('image_id:O', sort='-y'),
        y='area:Q',
        color='category_id:N',
        tooltip=['image_id', 'category_id', 'area']
    ).properties(
        title='Average Bounding Box Area per Image per Class (Sorted by Area)'
    ).interactive()

    chart.show()

    return (
        avg_area_per_image_per_class,
        avg_area_per_image_per_class_sorted,
        chart,
    )


if __name__ == "__main__":
    app.run()
