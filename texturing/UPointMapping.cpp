// UPointMapping.cpp
// Chao Du 2015 Jan
#include "UPointMapping.h"

void CalcHomographyFromPoints( const std::vector< cv::Vec3d > &nPtSrc,
                               const std::vector< cv::Vec3d > &nPtTgt,
                               cv::Mat &nH )
{
  assert( nPtSrc.size() == nPtTgt.size() );

  // set up the linear system
  // Ax = b
  int aNumRows = 3 * nPtSrc.size();
  int aNumCols = 9;
  cv::Mat aA( aNumRows, aNumCols, CV_64F );
  cv::Mat aB( aNumRows, 1, CV_64F );
  cv::Mat aSolution = cv::Mat::zeros( aNumRows, 1, CV_64F );

  for ( int i = 0; i < aNumRows / 3; ++i ) {

    // A
    aA.at< double >( i * 3, 0 ) = aA.at< double >( i * 3 + 1, 3 ) = aA.at< double >( i * 3 + 2, 6 ) = nPtSrc[ i ][ 0 ];
    aA.at< double >( i * 3, 1 ) = aA.at< double >( i * 3 + 1, 4 ) = aA.at< double >( i * 3 + 2, 7 ) = nPtSrc[ i ][ 1 ];
    aA.at< double >( i * 3, 2 ) = aA.at< double >( i * 3 + 1, 5 ) = aA.at< double >( i * 3 + 2, 8 ) = nPtSrc[ i ][ 2 ];

    // b
    aB.at< double >( i * 3    , 0 ) = nPtTgt[ i ][ 0 ];
    aB.at< double >( i * 3 + 1, 0 ) = nPtTgt[ i ][ 1 ];
    aB.at< double >( i * 3 + 2, 0 ) = nPtTgt[ i ][ 2 ];
  }

  // use svd to solve the system
  cv::Mat U, Vt, SIGMA( aNumCols, 1, CV_64F );
  cv::SVD::compute( aA, SIGMA, U, Vt );
  // U SIGMA V' x = b
  //   SIGMA V' x = U' b
  cv::Mat aTmpMat1 = U.t() * aB;
  // Y = SIGMA^-1 U' b (if SIGMA[i] == 0, we set Y[i] = 0)
  for ( int i = 0; i < std::min( aNumRows, aNumCols ); ++i ) {
    // REVISIT - check if SIGMA is nx1 or nxn
    if ( !( fabs( SIGMA.at< double >( i, 0 ) ) < 1e-6 ) ) {
      aSolution.at< double >( i, 0 ) = aTmpMat1.at< double >( i, 0 ) / SIGMA.at< double >( i, 0 );
    }
  }
  // X = V' Y
  aSolution = Vt.t() * aSolution;

  for ( int i = 0; i < aNumRows; ++i ) {
    nH.at< double >( i / 3, i % 3 ) = aSolution.at< double >( i, 0 );
  }
}

void CalcMappedPoints( const std::vector< cv::Vec3d > &nPtSrc,
                      std::vector< cv::Vec3d > &nPtTgt,
                      const cv::Mat &nH )
{
  if ( nPtTgt.size() != 0 ) {
    nPtTgt.clear();
  }

  for ( size_t i = 0; i < nPtSrc.size(); ++i ) {
    nPtTgt.push_back( CalcMappedPoint( nPtSrc[i], nH ) );
  }
}

cv::Vec3d CalcMappedPoint( const cv::Vec3d &ptSrc,
                           const cv::Mat   &homographyMatrix ) {

  cv::Mat aP( 3, 1, CV_64F );
  aP.at< double >( 0, 0 ) = ptSrc[ 0 ];
  aP.at< double >( 1, 0 ) = ptSrc[ 1 ];
  aP.at< double >( 2, 0 ) = ptSrc[ 2 ];

  aP = homographyMatrix * aP;

  cv::Vec3d ptTgt;
  ptTgt[0] = aP.at< double >( 0, 0 );
  ptTgt[1] = aP.at< double >( 1, 0 );
  ptTgt[2] = aP.at< double >( 2, 0 );

  return ptTgt;
}
