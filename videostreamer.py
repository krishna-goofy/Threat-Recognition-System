from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)

# Route to serve video files from the testing_videos folder
@app.route('/testing_videos/<path:filename>')
def testing_videos(filename):
    return send_from_directory('testing_videos', filename)

@app.route('/')
def index():
    # HTML with a video element that autoplays, loops, and has no controls.
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Looping Video</title>
        <style>
            /* Make the video fill the entire window */
            html, body {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background-color: black;
            }
            video {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <video autoplay loop muted playsinline>
            <source src="/testing_videos/V_19.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </body>
    </html>
    '''
    return render_template_string(html_content)

if __name__ == '__main__':
    # Binding to 0.0.0.0 makes the app accessible via your device's IP address.
    app.run(port=5005)
