// CWindow.cpp
// Chao Du 2014 Dec
#include "CWindow.h"

#include "HBase.h"

#include <QtCore>
#include <QtWidgets>

#include "volumepkg.h"
#include "CVolumeViewerWithCurve.h"

#include "UDataManipulateUtils.h"

#define _DEBUG

using namespace ChaoVis;


// Constructor
CWindow::CWindow( void ) :
    fVpkg( NULL ),
    fPathOnSliceIndex( 0 ),
    fVolumeViewerWidget( NULL ),
    fPathListWidget( NULL ),
    fPenTool( NULL ),
    fSegTool( NULL ),
    fWindowState( EWindowState::WindowStateIdle ),
    fSegmentationId( "" ),
    fMinSegIndex( VOLPKG_SLICE_MIN_INDEX ),
    fMaxSegIndex( VOLPKG_SLICE_MIN_INDEX )
{
    ui.setupUi( this );

    fVpkgChanged = false;

    // default parameters for segmentation method
    // REVISIT - refactor me
    fSegParams.fGravityScale = 0.3;
    fSegParams.fThreshold = 1;
    fSegParams.fEndOffset = 5;

    // create UI widgets
    CreateWidgets();

    // create menu
    CreateActions();
    CreateMenus();

    OpenSlice();
    UpdateView();

    update();
}

// Constructor with QRect windowSize
CWindow::CWindow( QRect windowSize ) :
        fVpkg( NULL ),
        fPathOnSliceIndex( 0 ),
        fVolumeViewerWidget( NULL ),
        fPathListWidget( NULL ),
        fPenTool( NULL ),
        fSegTool( NULL ),
        fWindowState( EWindowState::WindowStateIdle ),
        fSegmentationId( "" ),
        fMinSegIndex( VOLPKG_SLICE_MIN_INDEX ),
        fMaxSegIndex( VOLPKG_SLICE_MIN_INDEX )
{
    ui.setupUi( this );

    fVpkgChanged = false;

    int height = windowSize.height();
    int width = windowSize.width();

    //MIN DIMENSIONS
    window()->setMinimumHeight(height/2);
    window()->setMinimumWidth(width/2);
    //MAX DIMENSIONS
    window()->setMaximumHeight(height);
    window()->setMaximumWidth(width);

    // default parameters for segmentation method
    // REVISIT - refactor me
    fSegParams.fGravityScale = 0.3;
    fSegParams.fThreshold = 1;
    fSegParams.fEndOffset = 5;

    // create UI widgets
    CreateWidgets();

    // create menu
    CreateActions();
    CreateMenus();

    OpenSlice();
    UpdateView();

    update();
}

// Destructor
CWindow::~CWindow( void )
{
    deleteNULL( fVpkg );
}

// Handle mouse press event
void CWindow::mousePressEvent( QMouseEvent *nEvent )
{
}

// Handle key press event
void CWindow::keyPressEvent( QKeyEvent *event )
{
    if ( event->key() == Qt::Key_Escape ) {
        // REVISIT - should prompt warning before exit
        close();
    } else {
        // REVISIT - dispatch key press event
    }
}

// Create widgets
void CWindow::CreateWidgets( void )
{
    // add volume viewer
    QWidget *aTabSegment = this->findChild< QWidget * >( "tabSegment" );
    assert( aTabSegment != NULL );

    fVolumeViewerWidget = new CVolumeViewerWithCurve();

    QVBoxLayout *aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->addWidget( fVolumeViewerWidget );

    aTabSegment->setLayout( aWidgetLayout );

    // pass the reference of the curve to the widget
    fVolumeViewerWidget->SetSplineCurve( fSplineCurve );
    fVolumeViewerWidget->SetIntersectionCurve( fIntersectionCurve );

    connect( fVolumeViewerWidget, SIGNAL( SendSignalOnNextClicked() ), this, SLOT( OnLoadNextSlice() ) );
    connect( fVolumeViewerWidget, SIGNAL( SendSignalOnPrevClicked() ), this, SLOT( OnLoadPrevSlice() ) );
    connect( fVolumeViewerWidget, SIGNAL( SendSignalOnLoadAnyImage(int) ), this, SLOT( OnLoadAnySlice(int) ) );
    connect( fVolumeViewerWidget, SIGNAL( SendSignalPathChanged() ), this, SLOT( OnPathChanged() ) );

    // new path button
    QPushButton *aBtnNewPath = this->findChild< QPushButton * >( "btnNewPath" );
    QPushButton *aBtnRemovePath = this->findChild< QPushButton * >( "btnRemovePath" );
    aBtnRemovePath->setEnabled( false ); // Currently no methods from removing paths
    connect( aBtnNewPath, SIGNAL( clicked() ), this, SLOT( OnNewPathClicked() ) );

    // pen tool and edit tool
    fPenTool = this->findChild< QPushButton * >( "btnPenTool" );
    fSegTool = this->findChild< QPushButton * >( "btnSegTool" );
    connect( fPenTool, SIGNAL( clicked() ), this, SLOT( TogglePenTool() ) );
    connect( fSegTool, SIGNAL( clicked() ), this, SLOT( ToggleSegmentationTool() ) );

    // list of paths
    fPathListWidget = this->findChild< QListWidget * >( "lstPaths" );
    connect( fPathListWidget, SIGNAL( itemClicked( QListWidgetItem* ) ), this, SLOT( OnPathItemClicked( QListWidgetItem* ) ) );

    // segmentation methods
    QComboBox *aSegMethodsComboBox = this->findChild< QComboBox * >( "cmbSegMethods" );
    aSegMethodsComboBox->addItem( tr( "Structure Tensor Particle Simulation" ) );

    fEdtGravity = this->findChild< QLineEdit * >( "edtGravityVal" );
    fEdtSampleDist = this->findChild< QLineEdit * >( "edtSampleDistVal" );
    fEdtStartIndex = this->findChild< QLineEdit * >( "edtStartingSliceVal" );
    fEdtEndIndex = this->findChild< QLineEdit * >( "edtEndingSliceVal" );
    // REVISIT - consider switching to CSimpleNumEditBox
    // see QLineEdit doc to see the difference of textEdited() and textChanged()
    // doc.qt.io/qt-4.8/qlineedit.html#textEdited
    connect( fEdtGravity, SIGNAL( editingFinished() ), this, SLOT( OnEdtGravityValChange() ) );
    connect( fEdtSampleDist, SIGNAL( textEdited(QString) ), this, SLOT( OnEdtSampleDistValChange( QString ) ) );
    connect( fEdtStartIndex, SIGNAL( textEdited(QString) ), this, SLOT( OnEdtStartingSliceValChange( QString ) ) );
    connect( fEdtEndIndex, SIGNAL( editingFinished() ), this, SLOT( OnEdtEndingSliceValChange() ) );

    // start segmentation button
    QPushButton *aBtnStartSeg = this->findChild< QPushButton * >( "btnStartSeg" );
    connect( aBtnStartSeg, SIGNAL( clicked() ), this, SLOT( OnBtnStartSegClicked() ) );

    // Impact Range slider
    QSlider *fEdtImpactRange = this->findChild< QSlider * >( "sldImpactRange" );
    connect( fEdtImpactRange, SIGNAL( valueChanged(int) ), this, SLOT( OnEdtImpactRange( int ) ) );

    // Setup the status bar
    statusBar = this->findChild< QStatusBar * >( "statusBar" );
}

// Create menus
void CWindow::CreateMenus( void )
{
    fFileMenu = new QMenu( tr( "&File" ), this );
    fFileMenu->addAction( fOpenVolAct );
    fFileMenu->addAction( fSavePointCloudAct );
    fFileMenu->addSeparator();
    fFileMenu->addAction( fExitAct );

    fHelpMenu = new QMenu( tr( "&Help" ), this );
    fHelpMenu->addAction( fAboutAct );

    menuBar()->addMenu( fFileMenu );
    menuBar()->addMenu( fHelpMenu );
}

// Create actions
void CWindow::CreateActions( void )
{
    fOpenVolAct = new QAction( tr( "&Open volume..." ), this );
    connect( fOpenVolAct, SIGNAL( triggered() ), this, SLOT( Open() ) );
    fExitAct = new QAction( tr( "E&xit..." ), this );
    connect( fExitAct, SIGNAL( triggered() ), this, SLOT( Close() ) );
    fAboutAct = new QAction( tr( "&About..." ), this );
    connect( fAboutAct, SIGNAL( triggered() ), this, SLOT( About() ) );
    fSavePointCloudAct = new QAction( tr( "&Save volume..." ), this );
    connect( fSavePointCloudAct, SIGNAL( triggered() ), this, SLOT( SavePointCloud() ) );
}

// Asks User to Save Data Prior to VC.app Exit
void CWindow::closeEvent(QCloseEvent *closing)
{
    if ( SaveDialog() == SaveResponse::Continue ) {
        closing->accept();
    } else {
        closing->ignore();
    }
}

void CWindow::setWidgetsEnabled(bool state)
{
    this->findChild< QGroupBox * >( "grpVolManager" )->setEnabled( state );
    this->findChild< QGroupBox * >( "grpSeg" )->setEnabled( state );
    this->findChild< QPushButton *>( "btnSegTool" )->setEnabled( state );
    this->findChild< QPushButton *>( "btnPenTool" )->setEnabled( state );
    this->findChild< QGroupBox * >( "groupBox_4" )->setEnabled( state );
    fVolumeViewerWidget->setButtonsEnabled(state);
}

bool CWindow::InitializeVolumePkg( const std::string &nVpkgPath )
{
    deleteNULL( fVpkg );

    try {
        fVpkg = new VolumePkg( nVpkgPath );
    } catch(...) {
        std::cerr << "VC::Error: Volume package failed to initialize." << std::endl;
    }

    fVpkgChanged = false;

    if ( fVpkg == NULL ) {
        std::cerr << "VC::Error: Cannot open volume package at specified location: " << nVpkgPath << std::endl;
        QMessageBox::warning(this, "Error", "Volume package failed to load. Package might be corrupt.");
        return false;
    }

    return true;
}

CWindow::SaveResponse CWindow::SaveDialog( void ) {
    // Return if nothing has changed
    if ( !fVpkgChanged ) return SaveResponse::Continue;

    QMessageBox::StandardButton response = QMessageBox::question( this, "Save changes?",
                                                                  tr("Changes will be lost! Save volume package before continuing?\n"),
                                                                  QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel );
    switch (response) {
        case QMessageBox::Save:
            SavePointCloud();
            return SaveResponse::Continue;
        case QMessageBox::Discard:
            fVpkgChanged = false;
            return SaveResponse::Continue;
        case QMessageBox::Cancel:
            return SaveResponse::Cancelled;
        default:
            return SaveResponse::Cancelled; // should never be reached
    }
}

// Update the widgets
void CWindow::UpdateView( void )
{
    if ( fVpkg == NULL ) {
        setWidgetsEnabled(false);// Disable Widgets for User
        this->findChild< QLabel * >( "lblVpkgName" )->setText( "No Volume Package Loaded" );
        return;
    }

    setWidgetsEnabled(true);// Enable Widgets for User

    // show volume package name
    this->findChild< QLabel * >( "lblVpkgName" )->setText( QString( fVpkg->getPkgName().c_str() ) );

    // set widget accessibility properly based on the states: is drawing? is editing?
    fEdtGravity->setText( QString( "%1" ).arg( fSegParams.fGravityScale ) );
    fEdtSampleDist->setText( QString( "%1" ).arg( fSegParams.fThreshold ) );
    fEdtStartIndex->setText( QString( "%1" ).arg( fPathOnSliceIndex ) );
    
    if ( fSegParams.fEndOffset + fPathOnSliceIndex >= fVpkg->getNumberOfSlices() )
        fSegParams.fEndOffset = (fVpkg->getNumberOfSlices() - 1) - fPathOnSliceIndex;
    fEdtEndIndex->setText( QString( "%1" ).arg( fSegParams.fEndOffset + fPathOnSliceIndex ) ); // offset + starting index

    if ( fIntersectionCurve.GetPointsNum() == 0) { // no points in current slice
        fSegTool->setEnabled( false );
    } else {
        fSegTool->setEnabled( true );
    }

    if ( fSegmentationId.length() != 0 &&   // segmentation selected
         fMasterCloud.points.size() == 0 ) { // current cloud is empty
        fPenTool->setEnabled( true );
    } else {
        fPenTool->setEnabled( false );
    }

    // REVISIT - these two states should be mutually exclusive, we guarantee this when we toggle the button, BUGGY!
    if ( fWindowState == EWindowState::WindowStateIdle ) {
        fVolumeViewerWidget->SetViewState( CVolumeViewerWithCurve::EViewState::ViewStateIdle );
        this->findChild< QGroupBox * >( "grpVolManager" )->setEnabled( true );
        this->findChild< QGroupBox * >( "grpSeg" )->setEnabled( false );
    } else if ( fWindowState == EWindowState::WindowStateDrawPath ) {
        fVolumeViewerWidget->SetViewState( CVolumeViewerWithCurve::EViewState::ViewStateDraw );
        this->findChild< QGroupBox * >( "grpVolManager" )->setEnabled( false );
        this->findChild< QGroupBox * >( "grpSeg" )->setEnabled( false );
    } else if ( fWindowState == EWindowState::WindowStateSegmentation ) {
        fVolumeViewerWidget->SetViewState( CVolumeViewerWithCurve::EViewState::ViewStateEdit );
        this->findChild< QGroupBox * >( "grpVolManager" )->setEnabled( false );
        this->findChild< QGroupBox * >( "grpSeg" )->setEnabled( true ); // segmentation can be done only when seg tool is selected
    } else {
        // something else
    }

    fEdtStartIndex->setEnabled( false ); // starting slice is always the current slice
    fEdtSampleDist->setEnabled( false ); // currently we cannot let the user change the sample distance

    fVolumeViewerWidget->UpdateView();

    update();
}

// Activate a specific segmentation by ID
void CWindow::ChangePathItem( std::string segID ) {
    statusBar->clearMessage();

    // Close the current segmentation
    ResetPointCloud();

    // Activate requested segmentation
    fSegmentationId = segID;
    fVpkg->setActiveSegmentation( fSegmentationId );

    // load proper point cloud
    fMasterCloud = *fVpkg->openCloud();
    SetUpCurves();

    // Move us to the lowest slice index for the cloud
    fPathOnSliceIndex = fMinSegIndex;
    OpenSlice();
    SetCurrentCurve( fPathOnSliceIndex );

    UpdateView();
}

// Split fMasterCloud into fUpperCloud and fLowerCloud
void CWindow::SplitCloud( void )
{
    int aTotalNumOfImmutablePts = fMasterCloud.width * ( fPathOnSliceIndex - fMinSegIndex );
    for ( int i = 0; i < aTotalNumOfImmutablePts; ++i ) {
        fUpperPart.push_back( fMasterCloud.points[ i ] );
    }
    // resize so the parts can be concatenated
    fUpperPart.width = fMasterCloud.width;
    fUpperPart.height = fUpperPart.points.size() / fUpperPart.width;
    fUpperPart.points.resize( fUpperPart.width * fUpperPart.height );

    // lower part, the starting slice
    for ( int i = 0; i < fMasterCloud.width; ++i ) {
        if ( fMasterCloud.points[ i + aTotalNumOfImmutablePts ].z != -1 )
            fLowerPart.push_back( fMasterCloud.points[ i + aTotalNumOfImmutablePts ] );
    }

    if ( fLowerPart.width != fMasterCloud.width ){
        QMessageBox::information( this, tr( "Error" ), tr( "Starting chain length has null points. Try segmenting from an earlier slice." ) );
        CleanupSegmentation();
        return;
    }
}

// Do segmentation given the starting point cloud
void CWindow::DoSegmentation( void )
{
    statusBar->clearMessage();
    
    // REVISIT - do we need to get the latest value from the widgets since we constantly get the values?
    if ( !SetUpSegParams() ) {
        QMessageBox::information( this, tr( "Info" ), tr( "Invalid parameter for segmentation" ) );
        return;
    }

    // 2) do segmentation from the starting slice
    // how to create pcl::PointCloud::Ptr from a pcl::PointCloud?
    // stackoverflow.com/questions/10644429/create-a-pclpointcloudptr-from-a-pclpointcloud
    fLowerPart = volcart::segmentation::structureTensorParticleSim( pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >( fLowerPart )),
                                             *fVpkg,
                                             fSegParams.fGravityScale,
                                             fSegParams.fThreshold,
                                             fSegParams.fEndOffset );

    // 3) concatenate the two parts to form the complete point cloud
    fMasterCloud = fUpperPart + fLowerPart;
    fMasterCloud.width = fUpperPart.width;
    fMasterCloud.height = fMasterCloud.size() / fMasterCloud.width;

    statusBar->showMessage( tr("Segmentation complete") );
    fVpkgChanged = true;
}

void CWindow::CleanupSegmentation( void )
{
    fUpperPart.clear();
    fLowerPart.clear();
    fSegTool->setChecked( false );
    fWindowState = EWindowState::WindowStateIdle;
    SetUpCurves();
    OpenSlice();
    SetCurrentCurve( fPathOnSliceIndex );
}

// Set up the parameters for doing segmentation
bool CWindow::SetUpSegParams( void )
{
    bool aIsOk;

    // gravity value
    double aGravVal = fEdtGravity->text().toDouble( &aIsOk );
    if ( aIsOk ) {
        fSegParams.fGravityScale = aGravVal;
    } else {
        return false;
    }

    // sample distance
    int aNewVal = fEdtSampleDist->text().toInt( &aIsOk );
    if ( aIsOk ) {
        fSegParams.fThreshold = aNewVal;
    } else {
        return false;
    }

    // ending slice index
    aNewVal = fEdtEndIndex->text().toInt( &aIsOk );
    if ( aIsOk && aNewVal >= fPathOnSliceIndex && aNewVal < fVpkg->getNumberOfSlices() ) {
        fSegParams.fEndOffset = aNewVal - fPathOnSliceIndex; // difference between the starting slice and ending slice
    } else {
        return false;
    }

    return true;
}

// Get the curves for all the slices
void CWindow::SetUpCurves( void )
{
    if ( fVpkg == NULL || fMasterCloud.empty() ) {
        statusBar->showMessage( tr("Selected point cloud is empty") );
        std::cerr << "VC::Warning: Point cloud for this segmentation is empty." << std::endl;
        return;
    }
    fIntersections.clear();
    int minIndex, maxIndex;
    if ( fMasterCloud.points.size() == 0 ) {
        minIndex = maxIndex = fPathOnSliceIndex;
    } else {
        pcl::PointXYZRGB min_p, max_p;
        pcl::getMinMax3D( fMasterCloud, min_p, max_p );
        minIndex = floor( fMasterCloud.points[ 0 ].z );
        maxIndex = floor( max_p.z );
    }
    fMinSegIndex = minIndex;
    fMaxSegIndex = maxIndex;

    // assign rows of particles to the curves
    for ( int i = 0; i < fMasterCloud.height; ++i ) {
        CXCurve aCurve;
        for ( int j = 0; j < fMasterCloud.width; ++j ) {
            int pointIndex = j + (i * fMasterCloud.width);
            aCurve.SetSliceIndex((int) floor(fMasterCloud.points[pointIndex].z));
            aCurve.InsertPoint(Vec2<float>(fMasterCloud.points[pointIndex].x, fMasterCloud.points[pointIndex].y));
        }
        fIntersections.push_back( aCurve );
    }
}

// Set the current curve
void CWindow::SetCurrentCurve( int nCurrentSliceIndex )
{
    int curveIndex = nCurrentSliceIndex - fMinSegIndex;
    if (curveIndex >= 0 && curveIndex < fIntersections.size() && fIntersections.size() != 0) {
        fIntersectionCurve = fIntersections[curveIndex];
    }
    else {
        CXCurve emptyCurve;
        fIntersectionCurve = emptyCurve;
    }
}

// Open slice
void CWindow::OpenSlice( void )
{
    cv::Mat aImgMat;
    if ( fVpkg != NULL ) {
        fVpkg->volume().getSliceData( fPathOnSliceIndex ).copyTo( aImgMat );
        aImgMat.convertTo( aImgMat, CV_8UC3, 1.0 / 256.0 );
        cvtColor( aImgMat, aImgMat, CV_GRAY2BGR );
    } else
        aImgMat = cv::Mat::zeros(10, 10, CV_8UC3 );

    QImage aImgQImage;
    aImgQImage = Mat2QImage( aImgMat );
    fVolumeViewerWidget->SetImage( aImgQImage );
    fVolumeViewerWidget->SetImageIndex( fPathOnSliceIndex );
}

// Initialize path list
void CWindow::InitPathList( void ) {
    fPathListWidget->clear();
    if (fVpkg != NULL) {
        // show the existing paths
        for (size_t i = 0; i < fVpkg->getSegmentations().size(); ++i) {
            fPathListWidget->addItem(new QListWidgetItem(QString(fVpkg->getSegmentations()[i].c_str())));
        }
    }
}

// Update the Master cloud with the path we drew
void CWindow::SetPathPointCloud( void )
{
    // calculate the path and save that to aMasterCloud
    std::vector< cv::Vec2f > aSamplePts;
    fSplineCurve.GetSamplePoints( aSamplePts );

    pcl::PointXYZRGB point;
    pcl::PointCloud< pcl::PointXYZRGB > aPathCloud;
    for ( size_t i = 0; i < aSamplePts.size(); ++i ) {
        point.x = aSamplePts[ i ][ 0 ];
        point.y = aSamplePts[ i ][ 1 ];
        point.z = fPathOnSliceIndex;
        aPathCloud.push_back( point );
    }
    aPathCloud.width = aSamplePts.size();
    aPathCloud.height = 1;
    aPathCloud.resize( aPathCloud.width * aPathCloud.height );

    fMasterCloud = aPathCloud;
    fMinSegIndex = floor(fMasterCloud.points[0].z);
    fMaxSegIndex = fMinSegIndex;
}

// Open volume package
void CWindow::OpenVolume( void )
{
    QString aVpkgPath = QString( "" );
    aVpkgPath = QFileDialog::getExistingDirectory( this,
                                                   tr( "Open Directory" ),
                                                   QDir::homePath(),
                                                   QFileDialog::ShowDirsOnly |
                                                   QFileDialog::DontResolveSymlinks );
    // Dialog box cancelled
    if ( aVpkgPath.length() == 0 ) {
        std::cerr << "VC::Message: Open volume package cancelled." << std::endl;
        return;
    }

    // Checks the Folder Path for .volpkg extension
    std::string extension = aVpkgPath.toStdString().substr( aVpkgPath.toStdString().length() - 7, aVpkgPath.toStdString().length() );
    if ( extension.compare(".volpkg") != 0 ) {
        QMessageBox::warning(this, tr("ERROR"), "The selected file is not of the correct type: \".volpkg\"");
        std::cerr << "VC::Error: Selected file: " << aVpkgPath.toStdString() << " is of the wrong type." << std::endl;
        fVpkg = NULL; // Is need for User Experience, clears screen.
        return;
    }

    // Open volume package
    if ( !InitializeVolumePkg( aVpkgPath.toStdString() + "/" ) ) {
        return;
    }

    // Check version number
    if ( fVpkg->getVersion() < 2.0) {
        std::cerr << "VC::Error: Volume package is version " << fVpkg->getVersion() << " but this program requires a version >= 2.0." << std::endl;
        QMessageBox::warning( this, tr( "ERROR" ), "Volume package is version " + QString::number(fVpkg->getVersion()) + " but this program requires a version >= 2.0." );
        fVpkg = NULL;
        return;
    }

    fVpkgPath = aVpkgPath;
    fPathOnSliceIndex = 0;
}

void CWindow::CloseVolume( void ) {
    fVpkg = NULL;
    fSegmentationId = "";
    ResetPointCloud();
    OpenSlice();
    InitPathList();
    UpdateView();
}

// Reset point cloud
void CWindow::ResetPointCloud( void )
{
    fMasterCloud.clear();
    fUpperPart.clear();
    fLowerPart.clear();
    fIntersections.clear();
    CXCurve emptyCurve;
    fIntersectionCurve = emptyCurve;
}

// Handle open request
void CWindow::Open( void )
{
    if ( SaveDialog() == SaveResponse::Cancelled ) return;

    CloseVolume();
    OpenVolume();
    OpenSlice();
    InitPathList();
    UpdateView(); // update the panel when volume package is loaded
}

// Close application
void CWindow::Close( void )
{
    close();
}

// Pop up about dialog
void CWindow::About( void )
{
    // REVISIT - FILL ME HERE
    QMessageBox::information( this, tr( "About Volume Cartographer" ), tr( "Vis Center, University of Kentucky" ) );
}

// Save point cloud to path directory
void CWindow::SavePointCloud( void )
{
    if ( fMasterCloud.size() == 0 ) {
        std::cerr << "VC::message: Empty point cloud. Nothing to save." << std::endl;
        return;
    }

    // Try to save cloud to volpkg
    if ( fVpkg->saveCloud(fMasterCloud) != EXIT_SUCCESS ) {
        QMessageBox::warning(this, "Error", "Failed to write cloud to volume package.");
        return;
    }

    // Only mesh if we have more than one iteration of segmentation
    if (fMasterCloud.height <= 1) {
        std::cerr << "VC::message: Cloud height <= 1. Nothing to mesh." << std::endl;
    } else {
        if (fVpkg->saveMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>(fMasterCloud))) != EXIT_SUCCESS) {
            QMessageBox::warning(this, "Error", "Failed to write mesh to volume package.");
            return;
        } else {
            std::cerr << "VC::message: Succesfully saved mesh." << std::endl;
        }
    }

    statusBar->showMessage( tr("Volume saved.") , 5000 );
    std::cerr << "VC::message: Volume saved." << std::endl;
    fVpkgChanged = false;
}

// Create new path
void CWindow::OnNewPathClicked( void )
{
    // Save if we need to
    if ( SaveDialog() == SaveResponse::Cancelled ) return;

    // Make a new segmentation in the volpkg
    std::string newSegmentationId = fVpkg->newSegmentation();

    // add new path to path list
    QListWidgetItem *aNewPath = new QListWidgetItem( QString( newSegmentationId.c_str() ) );
    fPathListWidget->addItem( aNewPath );

    // Make sure we stay on the current slice
    fMinSegIndex = fPathOnSliceIndex;

    // Activate the new item
    fPathListWidget->setCurrentItem( aNewPath );
    ChangePathItem( newSegmentationId );
}

// Handle path item click event
void CWindow::OnPathItemClicked( QListWidgetItem *nItem )
{
    if ( SaveDialog() == SaveResponse::Cancelled ) {
        // Update the list to show the previous selection
        QListWidgetItem *previous = fPathListWidget->findItems( QString( fSegmentationId.c_str() ), Qt::MatchExactly )[0];
        fPathListWidget->setCurrentItem( previous );
        return;
    }

    ChangePathItem( nItem->text().toStdString() );
}

// Toggle the status of the pen tool
void CWindow::TogglePenTool( void )
{
    if ( fPenTool->isChecked() ) {
        fWindowState = EWindowState::WindowStateDrawPath;

        // turn off edit tool
        fSegTool->setChecked( false );
    } else {
        fWindowState = EWindowState::WindowStateIdle;

        if ( fSplineCurve.GetNumOfControlPoints() > 2 ) {
            SetPathPointCloud(); // finished drawing, set up path
            SavePointCloud();
            SetUpCurves();
            OpenSlice();
            SetCurrentCurve( fPathOnSliceIndex );
            fSplineCurve.Clear();
            fVolumeViewerWidget->ResetSplineCurve();
        }
    }

    UpdateView();
}

// Toggle the status of the segmentation tool
void CWindow::ToggleSegmentationTool( void )
{
    if ( fSegTool->isChecked() ) {
        fWindowState = EWindowState::WindowStateSegmentation;
        fUpperPart.clear();
        fLowerPart.clear();
        SplitCloud();

        // turn off edit tool
        fPenTool->setChecked( false );
    } else {
        CleanupSegmentation();
    }
    UpdateView();
}

// Handle gravity value change
void CWindow::OnEdtGravityValChange()
{
    bool aIsOk;
    double aNewVal = fEdtGravity->text().toDouble( &aIsOk );
    if ( aIsOk ) {
        if ( aNewVal <= 0.0 ) {
            aNewVal = 0.1;
            fEdtGravity->setText(QString::number(aNewVal));
        }
        else if ( aNewVal > 0.8 ) {
            aNewVal = 0.8;
            fEdtGravity->setText(QString::number(aNewVal));
        }
        fSegParams.fGravityScale = aNewVal;
    }
}

// Handle sample distance value change
void CWindow::OnEdtSampleDistValChange( QString nText )
{
    // REVISIT - the widget should be disabled and the change ignored for now
    bool aIsOk;
    int aNewVal = nText.toInt( &aIsOk );
    if ( aIsOk ) {
        fSegParams.fThreshold = aNewVal;
    }
}

// Handle starting slice value change
void CWindow::OnEdtStartingSliceValChange( QString nText )
{
    // REVISIT - FILL ME HERE
    // REVISIT - should be equivalent to "set current slice", the same as navigation through slices
}

// Handle ending slice value change
void CWindow::OnEdtEndingSliceValChange()
{
    // ending slice index
    bool aIsOk = false;
    int aNewVal = fEdtEndIndex->displayText().toInt( &aIsOk );
    if ( aIsOk && aNewVal > fPathOnSliceIndex && aNewVal < fVpkg->getNumberOfSlices() ) {
        fSegParams.fEndOffset = aNewVal - fPathOnSliceIndex; // difference between the starting slice and ending slice
    } else {
        statusBar->showMessage( tr("ERROR: Selected slice is out of range of the volume!"), 10000 );
        fEdtEndIndex->setText( QString::number(fPathOnSliceIndex + fSegParams.fEndOffset) );
    }
}

// Handle start segmentation
void CWindow::OnBtnStartSegClicked( void )
{
    DoSegmentation();
    CleanupSegmentation();
    UpdateView();
}

// Handle start segmentation
void CWindow::OnEdtImpactRange( int nImpactRange )
{
    fVolumeViewerWidget->SetImpactRange( nImpactRange );
}

// Handle loading any slice
void CWindow::OnLoadAnySlice( int nSliceIndex )
{
    if ( nSliceIndex >= 0 && nSliceIndex < fVpkg->getNumberOfSlices() ) {
        fPathOnSliceIndex = nSliceIndex;
        OpenSlice();
        SetCurrentCurve( fPathOnSliceIndex );
        UpdateView();
    } else
        statusBar->showMessage( tr("ERROR: Selected slice is out of range of the volume!"), 10000 );
}

// Handle loading the next slice
void CWindow::OnLoadNextSlice( void )
{
    if (fPathOnSliceIndex < fVpkg->getNumberOfSlices() - 1) {
        ++fPathOnSliceIndex;
        OpenSlice();
        SetCurrentCurve(fPathOnSliceIndex);
        UpdateView();
    } else
        statusBar->showMessage( tr("Already at the end of the volume!"), 10000 );
}

// Handle loading the previous slice
void CWindow::OnLoadPrevSlice( void )
{
    if (fPathOnSliceIndex > 0) {
        --fPathOnSliceIndex;
        OpenSlice();
        SetCurrentCurve(fPathOnSliceIndex);
        UpdateView();
    } else
        statusBar->showMessage( tr("Already at the beginning of the volume!"), 10000 );
}

// Handle path change event
void CWindow::OnPathChanged( void )
{
    if (fWindowState == EWindowState::WindowStateSegmentation) {
        // update current slice
        fLowerPart.clear();
        for (size_t i = 0; i < fIntersectionCurve.GetPointsNum(); ++i) {
            pcl::PointXYZRGB tempPt;
            tempPt.x = fIntersectionCurve.GetPoint(i)[0];
            tempPt.y = fIntersectionCurve.GetPoint(i)[1];
            tempPt.z = fPathOnSliceIndex;
            fLowerPart.push_back(tempPt);
        }
    }
}