#include "ofApp.h"

const string allowed_ext[] = {"jpg", "png", "gif", "jpeg"};

void ofApp::scan_dir_imgs(ofDirectory dir){
    ofDirectory new_dir;
    int size = dir.listDir();
    for (int i = 0; i < size; i++){
        if (dir.getFile(i).isDirectory()){
            new_dir = ofDirectory(dir.getFile(i).getAbsolutePath());
            new_dir.listDir();
            new_dir.sort();
            scan_dir_imgs(new_dir);
        } else if (std::find(std::begin(allowed_ext),
                             std::end(allowed_ext),
                             dir.getFile(i).getExtension()) != std::end(allowed_ext)) {
            imageFiles.push_back(dir.getFile(i));
        }
    }
}

// Returns a pair of grid side lengths that describe the grid
// closest to a square that contains n_tiles items
std::tuple<int, int> ofApp::best_grid_size(int n_tiles) {
    
    int grid_x = 0;
    int grid_y = 0;
    
    for(int n = 1; n < ceil(sqrt(n_tiles) + 1); n++) {
        if(n_tiles % n == 0) {
            grid_y = n;
            grid_x = n_tiles/n;
        }
    }
    
    return std::tuple<int, int>(grid_x, grid_y);
}

void save_tsne_to_json(vector<ofFile> image_files,
                       int nx, int ny,
                       vector<ofVec2f> tsne_points,
                       vector<ofVec2f> solved_grid,
                       ofFile out_file) {
    
    ofFile new_out_file(out_file, ofFile::WriteOnly);
    
    if(!new_out_file.exists()) {
        new_out_file.create();
    }
    
    Json::Value images(Json::arrayValue);
    
    for(int i = 0; i < solved_grid.size(); i++) {
        Json::Value image;
        image["filename"] = image_files[i].getFileName();
        
        Json::Value tsne_pos;
        tsne_pos["x"] = tsne_points[i].x;
        tsne_pos["y"] = tsne_points[i].y;
        
        image["tsne_pos"] = tsne_pos;
        
        Json::Value grid_pos;
        grid_pos["x"] = round(solved_grid[i].x * nx);
        grid_pos["y"] = round(solved_grid[i].y * ny);
        
        image["grid_pos"] = grid_pos;
        
        images.append(image);
    }
    
    Json::StyledWriter writer;
    new_out_file << writer.write(images);
    new_out_file.close();
}

//--------------------------------------------------------------
void ofApp::setup(){
    
    // SETUP
    // imageDir, imageSavePath = location of images, path to save the final grid image
    // n_images = the number of images to use
    // nx, ny = size of the grid, computed to make the grid closest to a
    // square
    // w, h = downsample (or scale up) for source images prior to encoding!
    // displayW, displayH = resolution of the individual thumbnails for your output image - be careful about going over your maximum texture size on graphics card - 5000x5000 may work, but 10000x10000 may not
    // above these sizes, you need to manually save the image as binary data
    // perplexity, theta (for t-SNE, see 'example' for explanation of these)
    string imageDir = "/Users/a/Pictures/inspires";
    string imageSavePath = "tsne_grid_insp.png";
    string results_save_json = "tsne_grid_insp.json";
    int n_images = 300;
    
    int nx, ny;
    std::tie(nx, ny) = best_grid_size(n_images);

    w = 256; //do not go lower than 256 - it will work, but results won't be as good
    h = 256;
    displayW = 100;
    displayH = 100;
    perplexity = 75;
    theta = 0.2;

    
    /////////////////////////////////////////////////////////////////////
    // CCV activations -> t-SNE embedding -> grid assignments
    
    // get images recursively from directory
    ofLog() << "Gathering images...";
    ofDirectory dir = ofDirectory(imageDir);
    scan_dir_imgs(dir);
    if (imageFiles.size() < nx * ny) {
        ofLog(OF_LOG_ERROR, "There are less images in the directory than the grid size requested (nx*ny="+ofToString((nx*ny))+"). Exiting to save you trouble...");
        ofExit(); // not enough images to fill the grid, so quitting
    }
    
    // load all the images
    for(int i = 0; i < nx * ny; i++) {
        if (i % 20 == 0)    ofLog() << " - loading image "<<i<<" / "<<nx*ny<<" ("<<dir.size()<<" in dir)";
        images.push_back(ofImage());
        images.back().load(imageFiles[i]);
    }

    // resize images to w x h
    for (int i = 0; i < images.size(); i++) {
        if (images[i].getWidth() > images[i].getHeight()) {
            images[i].crop((images[i].getWidth()-images[i].getHeight()) * 0.5, 0, images[i].getHeight(), images[i].getHeight());
        }
        else if (images[i].getHeight() > images[i].getWidth()) {
            images[i].crop(0, (images[i].getHeight()-images[i].getWidth()) * 0.5, images[i].getWidth(), images[i].getWidth());
        }
        images[i].resize(w, h);
    }
    
    // setup ofxCcv
    ccv.setup("image-net-2012.sqlite3");
    
    // encode all of the images with ofxCcv
    ofLog() << "Encoding images...";
    for (int i=0; i<images.size(); i++) {
        if (i % 20 == 0) ofLog() << " - encoding image "<<i<<" / "<<images.size();
        vector<float> encoding = ccv.encode(images[i], ccv.numLayers()-1);
        encodings.push_back(encoding);
    }
    
    // run t-SNE and load image points to imagePoints
    ofLog() << "Run t-SNE on images";
    tsneVecs = tsne.run(encodings, 2, perplexity, theta, true);
    
    // solve assignment grid
    vector<ofVec2f> tsnePoints; // convert vector<double> to ofVec2f
    for (auto t : tsneVecs) tsnePoints.push_back(ofVec2f(t[0], t[1]));
    vector<ofVec2f> gridPoints = makeGrid(nx, ny);
    solvedGrid = solver.match(tsnePoints, gridPoints, false);
    
    // save tsne_results
    ofFile out_file(results_save_json, ofFile::WriteOnly);
    save_tsne_to_json(imageFiles, nx, ny, tsnePoints, solvedGrid, out_file);
    
    // save images
    ofFbo fbo;
    fbo.allocate(nx * displayW, ny * displayH);
    fbo.begin();
    ofClear(0, 0);
    ofBackground(0);
    for (int i=0; i<solvedGrid.size(); i++) {
        float x = (fbo.getWidth() - displayW) * solvedGrid[i].x;
        float y = (fbo.getHeight() - displayH) * solvedGrid[i].y;
        images[i].draw(x, y, displayW, displayH);
    }
    fbo.end();
    ofImage img;
    fbo.readToPixels(img);
    img.save(imageSavePath);
    
    // setup gui
    gui.setup();
    gui.add(scale.set("scale", 1.0, 0.0, 1.0));
}

//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(0);
    
    ofPushMatrix();
    ofTranslate(-ofGetMouseX() * (scale - 0.5), -ofGetMouseY() * (scale - 0.5));
    for (int i=0; i < solvedGrid.size(); i++) {
        float x = scale * (nx - 1) * w * solvedGrid[i].x;
        float y = scale * (ny - 1) * h * solvedGrid[i].y;
        images[i].draw(x, y, scale * images[i].getWidth(), scale * images[i].getHeight());
    }
    ofPopMatrix();
    
    gui.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}
