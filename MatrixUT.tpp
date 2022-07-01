#ifndef __ML_MATRIX_TPP
#define __ML_MATRIX_TPP


/***************************************************************************************************
  Constructors
*/
// Create and fill a matrix (with 0s if not specified)
template<typename T>
MLMatrix<T>::MLMatrix(unsigned _rows, unsigned _cols, const T initial) {
  rows = _rows;
  cols = _cols;
  mat.resize(rows);
  for (unsigned i = 0; i < rows; ++i) {
    mat[i].resize(cols, initial);
  }
}

// random matrix Constructor
template<typename T> MLMatrix<T>::MLMatrix(unsigned _rows, unsigned _cols, const T min, const T max)
{
  rows = _rows;
  cols = _cols;
  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) {
    mat[i].resize(cols);
    for (size_t j = 0; j < cols; ++j) {
      if (std::is_floating_point<T>::value) { // float
        float x = esp_random();
        x /= UINT32_MAX;
        mat[i][j] = min + x * (max - min);
      } else {
        mat[i][j] = random(min, max);
      }
    }
  }
}

// Create a new matrix of the same size as another matrix, filled with constant (default 0)
template<typename T> template<typename U> MLMatrix<T>::MLMatrix(MLMatrix<U> const &rhs, const T initial)
{
  rows = rhs.get_rows();
  cols = rhs.get_cols();
  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) {
    mat[i].resize(cols, initial);
  }
}

// Copy Constructor (from a vector)
template<typename T> template<typename U> MLMatrix<T>::MLMatrix(const std::vector<U>& rhs) {
  rows = rhs.size();
  cols = 1;
  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) mat[i].resize(cols, static_cast<T>(rhs[i]));
}

// Copy Constructor (from a vector of vectors, each vector is a new row)
template<typename T> template<typename U> MLMatrix<T>::MLMatrix(const std::vector<std::vector<U> >& rhs) {
  rows = rhs.size();
  cols = rhs[0].size();

  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) {
    mat[i].resize(cols);
    for (size_t j = 0; j < cols; ++j) mat[i][j] = static_cast<T>(rhs[i][j]);
  }
}

// Copy Constructor (from an array)
template<typename T> template<typename U> MLMatrix<T>::MLMatrix(const U rhs[], const unsigned dim) {
  rows = dim;
  cols = 1;
  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) mat[i].resize(cols, static_cast<T>(rhs[i]));
}

// (Virtual) Destructor
template<typename T>
MLMatrix<T>::~MLMatrix() {}

/***************************************************************************************************
  Accessors
*/
// Access the individual elements
template<typename T>
typename std::vector<T>::reference MLMatrix<T>::operator()(const unsigned& row, const unsigned& col) {
  return mat[row][col];
}

// Access the individual elements (const)                                                                                                                                     
template<typename T>
T const MLMatrix<T>::operator()(const unsigned& row, const unsigned& col) const {
  return mat[row][col];
}


/***************************************************************************************************
  Initialization methods
*/

// Assignment Operator
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator=(const MLMatrix<U>& rhs) {
  rows = rhs.get_rows();
  cols = rhs.get_cols();

  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) {
    mat[i].resize(cols);
    for (size_t j = 0; j < cols; ++j) {
      mat[i][j] = static_cast<T>(rhs(i, j));
    }
  }
  return *this;
}

// Assignment from vector
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator=(const std::vector<U>& rhs) {
  rows = rhs.size();
  cols = 1;

  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) {
    mat[i].resize(cols);
    mat[i][0] = static_cast<T>(rhs[i]);
  }
return *this;
}

// Assignment from vector of vectors
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator=(const std::vector<std::vector<U> >& rhs) {
  rows = rhs.size();
  cols = rhs[0].size();

  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) {
    mat[i].resize(cols);
    for (size_t j = 0; j < cols; ++j) mat[i][j] = static_cast<T>(rhs[i][j]);
  }
  return *this;
}

/* Assignment from array. Usage:
  float R[5] = {1,2,3,4,5};
  MLMatrix<uint8_t> S;
  S.fromArray(R, 5);
*/
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::fromArray(const U rhs[], const unsigned dim) {
  rows = dim;
  cols = 1;
  mat.resize(rows);
  for (size_t i = 0; i < rows; ++i) mat[i].resize(cols, static_cast<T>(rhs[i]));
  return *this;
}


/***************************************************************************************************
  Overloaded operators
*/

// Cumulative addition of this matrix and another
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator+=(const MLMatrix<U>& rhs) {
  if ( rows != rhs.get_rows() || cols != rhs.get_cols() ) { // matrices of different sizes
    Serial.printf("Addition error: dimensions do not match (%d, %d)+(%d, %d)", rows, cols, rhs.get_rows()), rhs.get_cols();
    while(1);
  }

  MLMatrix result = (*this) + rhs;
  (*this) = result;
  return *this;
}

// Cumulative substraction of this matrix and another
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator-=(const MLMatrix<U>& rhs) {
  if ( rows != rhs.get_rows() || cols != rhs.get_cols() ) { // matrices of different sizes
    Serial.printf("Addition error: dimensions do not match (%d, %d)-(%d, %d)", rows, cols, rhs.get_rows()), rhs.get_cols();
    while(1);
  }

  MLMatrix result = (*this) - rhs;
  (*this) = result;
  return *this;
}

// Cumulative left multiplication of this matrix and another
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator*=(const MLMatrix<U>& rhs) {
  if (cols != rhs.get_rows()) {
    Serial.printf("Multiplication error: dimensions do not match (%d, %d).(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  MLMatrix result = (*this) * rhs;
  (*this) = result;
  return *this;
}

// Matrix/scalar addition
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator+=(const U& rhs) {
  MLMatrix result = (*this) + rhs;
  (*this) = result;
  return *this;
}

// Matrix/scalar substraction
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator-=(const U& rhs) {
  MLMatrix result = (*this) - rhs;
  (*this) = result;
  return *this;
}

// Matrix/scalar multiplication
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator*=(const U& rhs) {
  MLMatrix result = (*this) * rhs;
  (*this) = result;
  return *this;
}

// Matrix/scalar division
template<typename T> template<typename U> MLMatrix<T>& MLMatrix<T>::operator/=(const U& rhs) {
  MLMatrix result = (*this) / rhs;
  (*this) = result;
  return *this;
}

/***************************************************************************************************
  Comparison operators
*/
// Determine if two matrices are equal and return true, otherwise return false.
template <typename T> template<typename U> 
const bool MLMatrix<T>::operator==(const MLMatrix<U> &rhs) const
{
  if ( rows != rhs.get_rows() || cols != rhs.get_cols() ) return false; // matrices of different sizes

  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      if (mat[i][i] != rhs(i,j)) return false; 
  return true; // matrices are equal
}

template <typename T> template<typename U> const bool MLMatrix<T>::operator!=(const MLMatrix<U> &rhs) const
{
  return !(*this == rhs);
}

// Compare 2 matrices element wise
template <typename T> template<typename U> MLMatrix<bool> MLMatrix<T>::operator<(const MLMatrix<U> &rhs)
{
  if (rows != rhs.get_rows() || cols != rhs.get_cols()) { // matrices of different sizes
    Serial.printf("Comparison error: dimensions do not match (%d, %d)<(%d, %d)", rows, cols, rhs.get_rows()), rhs.get_cols();
    while(1);
  }
  MLMatrix<bool> result(rows, cols, 0);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      result(i,j) = (mat[i][j] < rhs(i,j)) ? true : false;
  return result;
}

template <typename T> template<typename U> MLMatrix<bool> MLMatrix<T>::operator>=(const MLMatrix<U> &rhs)
{
  if ( rows != rhs.get_rows() || cols != rhs.get_cols() ) { // matrices of different sizes
    Serial.printf("Comparison error: dimensions do not match (%d, %d)>=(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }
  MLMatrix<bool> result(rows, cols, 0);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      result(i,j) = (mat[i][j] >= rhs(i,j)) ? true : false; 
  return result;
}


/***************************************************************************************************
  Operations on matrices
*/
// Calculate a transpose of this matrix
template<typename T>
MLMatrix<T> MLMatrix<T>::transpose() {
  MLMatrix result(cols, rows, 0);

  for (size_t i = 0; i < cols; ++i) {
    for (size_t j = 0; j < rows; ++j) {
      result(i,j) = mat[j][i];
    }
  }
  return result;
}

// Return a matrix with elemnts squared: M = A.square()
template<typename T>
MLMatrix<T> MLMatrix<T>::square ()
{
  MLMatrix result(rows, cols, 0);

  for (size_t i=0; i<rows; ++i) {
    for (size_t j=0; j<cols; ++j) {
      result(i,j) = pow(mat[i][j], 2);
    }
  }
  return result;
}

/* 
  Hadamard (element-wise) product
  Usage:
    MLMatrix<int> a(10, 50, 0, 10); // define the first matrix
    MLMatrix<int> b(10, 50, 0, 10); // define the second matrix, same dimensions
    a = a.Hadamard(b);

*/
template<typename T> template<typename U> MLMatrix<T> MLMatrix<T>::Hadamard(const MLMatrix<U>& rhs, bool clip)
{
  if ( rows != rhs.get_rows() || cols != rhs.get_cols() ) { // matrices of different sizes
    Serial.printf("Hadamard product error: dimensions do not match (%d, %d).(%d, %d)", rows, cols, rhs.get_rows(), rhs.get_cols());
    while(1);
  }
  MLMatrix<T> result(rows, cols, 0);

  if (!clip) {
    for ( size_t i = 0; i < rows; ++i )
      for ( size_t j = 0; j < cols; ++j )
        result(i,j) = mat[i][j] * static_cast<T>(rhs(i,j));
  } else {
    for ( size_t i = 0; i < rows; ++i )
      for ( size_t j = 0; j < cols; ++j ) {
        float R = (float)mat[i][j] * (float)rhs(i,j);
        if (R < std::numeric_limits<T>::min()) R = std::numeric_limits<T>::min();
        if (R > std::numeric_limits<T>::max()) R = std::numeric_limits<T>::max();
        result(i,j) = T(R);      
      }
  }
  return result;
}

/***************************************************************************************************
  Matrix vector operations
*/

// Obtain a vector of the diagonal elements
// Usage :  std::vector<int> Diag = mat.diag_vec();
template<typename T>
std::vector<T> MLMatrix<T>::diag_vec() {
  std::vector<T> result(rows, 0);

  for (unsigned i=0; i<rows; ++i) result[i] = mat[i][i];
  return result;
}

// Cumulative multiplication of a matrix by a vector
// Cumulative left multiplication of this matrix and another
template<typename T> template<typename U> std::vector<T>& MLMatrix<T>::operator*=(const std::vector<U>& rhs) {
  std::vector<T> result = (*this) * rhs;
  (*this) = result;
  return *this;
}


/***************************************************************************************************
  Vector operations
*/

// Vector dot product, using matrices
/*
Example usage:
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  MLMatrix<int> mv1(3, 1, 0);
  mv1 = v1;
  MLMatrix<int> mv2(3, 1, 0);
  mv2 = v2;
  int P = mv1.MdotProd(mv2, true);
*/
template<typename T> template<typename U> auto MLMatrix<T>::MdotProd(const MLMatrix<U>& rhs, bool clip) ->
    decltype(std::declval<U>()*std::declval<T>())
{
    using ret_t = decltype(std::declval<U>()*std::declval<T>());
    if (rhs.get_cols() !=1 || cols != 1) {
      Serial.printf("Dot product error: please use matrices with 1 column\n");
      while(1);
    }
    if (rhs.get_rows() != rows) {
        Serial.printf("Dot product error: dimensions do not match (%d, %d)\n",rhs.get_rows(), rows);
        while(1);
    }
    double sum = 0.0f;
    for (unsigned i = 0; i < rows; ++i) sum += mat[i][0] * rhs(i, 0);

    if (clip) { // Clip the result at min and max value of type T
        // float sum = std::inner_product(std::begin(a.mat), std::end(a.mat), std::begin(mat), 0.0);;
        if (sum < std::numeric_limits<ret_t>::min()) sum = std::numeric_limits<ret_t>::min();
        if (sum > std::numeric_limits<ret_t>::max()) sum = std::numeric_limits<ret_t>::max();
    }
    return static_cast<ret_t>(sum);
}


/***************************************************************************************************
  Norms
*/

/* Various normS of a matrix
  Usage:
    MLMatrix<float> a(10, 50, 0.0f, 10.0f);
    float normL0 = a.L0Norm();
    float normL1 = a.L1Norm();
    float normL2 = a.L2Norm();
*/
template<typename T>
T MLMatrix<T>::L2Norm()
{
  T L2 = T(0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      L2 += pow(this->mat[i][j], 2);
    }
  }
  return sqrt(L2);
}

template<typename T>
T MLMatrix<T>::L1Norm()
{
  T L1 = T(0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      float L = abs(this->mat[i][j]);
      L1 = (L > L1)? L: L1;
    }
  }
  return L1;
}

template<typename T>
int MLMatrix<T>::L0Norm() // number of non zero elements
{
  int L0 = 0;
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (this->mat[i][j] != 0) ++L0;
    }
  }
  return L0;
}

// Max, min, mean, std deviation
template<typename T>
T MLMatrix<T>::max() const
{
  T max = std::numeric_limits<T>::min();
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (this->mat[i][j] > max) max = this->mat[i][j];
    }
  }
  return max;
}

template<typename T>
T MLMatrix<T>::min() const
{
  T min = std::numeric_limits<T>::max();
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (this->mat[i][j] < min) min = this->mat[i][j];
    }
  }
  return min;
}

template<typename T>
float MLMatrix<T>::mean() const
{
  float mean = 0.0f;
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      mean += this->mat[i][j];
    }
  }
  mean /= float(rows * cols);
  return mean;
}

// Compute the mean absolute value of a row
template<typename T>
float MLMatrix<T>::meanRow(int rowNumber)
{
  float mean = 0.0f;
  for (unsigned j=0; j<cols; ++j) mean += abs(this->mat[rowNumber][j]);
  mean /= float(cols);
  return mean;
}

template<typename T>
float MLMatrix<T>::stdev(const float mean) const
{
  float stdev = 0.0f;
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      stdev += pow(this->mat[i][j] - mean, 2);
    }
  }
  stdev /= float(rows * cols);
  stdev = sqrt(stdev);
  return stdev;
}

// Find the place of the minimum value
template <typename T>
void MLMatrix<T>::indexMin(int &indexRow, int &indexCol)
{
  T minVal = this->mat[0][0];
    for (int i = 0; i < rows; ++i) 
      for (int j = 0; j < cols; ++j)
        if (this->mat[i][j] < minVal) {
          minVal = this->mat[i][j];
          indexRow = i;
          indexCol = j;
      }
}

// Find the place of the maximum value
template <typename T>
void MLMatrix<T>::indexMax(int &indexRow, int &indexCol)
{
  indexRow = 0;
  indexCol = 0;
  T maxVal = this->mat[0][0];
  for (int i = 0; i < rows; ++i) 
    for (int j = 0; j < cols; ++j)
      if (this->mat[i][j] > maxVal) {
        maxVal = this->mat[i][j];
        indexRow = i;
        indexCol = j;
      }
}


/***************************************************************************************************
  Misc methods
*/

// Get the number of rows of the matrix
template<typename T>
unsigned MLMatrix<T>::get_rows() const { return this->rows; }

// Get the number of columns of the matrix                                                                                                                                    
template<typename T>
unsigned MLMatrix<T>::get_cols() const { return this->cols; }

template <typename T>
void MLMatrix<T>::setSize(const int _rows, const int _cols, const T val)
{
  rows = _rows;
  cols = _cols;
  mat.resize(_rows);
  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i].resize(_cols, val);
  }
}

// Display the matrix
// usage: mat.print();
template <typename T>
void MLMatrix<T>::print()
{
  Serial.printf("%d rows, %d cols\n",rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    if (std::is_floating_point<T>::value) { // float
      for (size_t j = 0; j < cols - 1; ++j) Serial.printf("%8.3f, ", this->mat[i][j]);
        Serial.printf("%8.3f\n", this->mat[i][cols - 1]);
    }  else { // integer
      for (size_t j = 0; j < cols - 1; ++j) Serial.printf("%5d, ", this->mat[i][j]);
      Serial.printf("%5d\n", this->mat[i][cols - 1]);
    }  
  }
}

// Specific function for boolean matrices
template <typename T>
void MLMatrix<T>::printBool()
{
  Serial.printf("%d rows, %d cols\n",rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols - 1; ++j)
      if (mat[i][j]) Serial.print("true, "); 
      else Serial.print("false, ");
    if (mat[i][cols - 1]) Serial.println("true"); 
    else Serial.println("false");
  }
}

// Display the matrix size
// usage: mat.printSize();
template <typename T>
void MLMatrix<T>::printSize()
{
  Serial.printf("(%d, %d)",rows, cols);
}

/*
  Apply a given function to the elements of a matrix (element-wise)
  The function must be written as: 
    T function(T x) { ... }
  For example :
    int plusOne (int x) { return x+1; }
  
  Usage:
    MLMatrix<int> a(10, 50, 0, 10); // define the first matrix
    a.applySelf( &function ); // changes the matrix
    MLMatrix<int> b = a.apply( &function ); // does not change the matrix
*/
template<typename T>
MLMatrix<T> MLMatrix<T>::applySelf(T (*function)(T))
{
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      mat[i][j] = function(mat[i][j]);
    }
  }
  return *this;
}

template<typename T>
MLMatrix<T> MLMatrix<T>::apply(T (*function)(T)) 
{
  MLMatrix result(rows, cols, 0);

  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      result(i,j) = function(this->mat[i][j]);
    }
  }
  return result;
}

// Apply a random change to all elements of a matrix
// Example : randomChange(0.1) applies random multiplication by a factor in [0.9, 1.1]
template<typename T>
MLMatrix<T> MLMatrix<T>::randomChange(const float amplitude)
{
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      // random number between -1 and +1
      float rand = float(random(10000)) / 10000.0f * 2.0f - 1.0f; // random in [-1, +1]
      mat[i][j] = mat[i][j] * (1.0f + rand * amplitude);
    }
  }
  return *this;
}

/*  Generate a random matrix with normal distribution
    using the polar form of Box Muller algorithm
    usage:    MLMatrix<float> u(30, 30, 0.0f);
              u.randomNormal(0.0f,1.0f);
*/
template<typename T>
MLMatrix<T> MLMatrix<T>::randomNormal(const float mean, float std_dev)
{
  std_dev = abs(std_dev);
  int dim = rows * cols;
  float eps = 0.001f;
  MLMatrix<T> N(rows, cols, 0);
  MLMatrix<T> C(dim, 1, 0.0f);
  for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j) {
      float r = 1.0f;
      float u;
      do {
        u = 2 * float(random(100000)) / 100000.0f - 1;
        float v = 2 * float(random(100000)) / 100000.0f - 1;
        r = u * u + v * v;
      } while (r > 1.0f && r != 0.0f);
      mat[i][j] = u * sqrt(-2.0f * log(r) / r);
      mat[i][j] = mat[i][j] * std_dev - mean;   
    }
  return *this;
}


// Scale the norm to a given value
// usage: bool zeroNorm = X.normScale2(val);
// Returns true if L2 norm is zero, else false
template <typename T>
bool MLMatrix<T>::normScale2 (float value)
{
  bool zeroNorm = false;
  value = abs(value);
  float L2 = this->L2Norm();
  if (L2 == 0.0f) zeroNorm = true; // don't scale if L2 norm is zero
  else {
    float coef = value / L2;
    for (unsigned i=0; i<rows; ++i)
      for (unsigned j=0; j<cols; ++j)
        mat[i][j] *= coef;
  }
  return zeroNorm;
}

// Clip all values less than threshold to zero
// Leads to:  |abs(value)| > threshold or zero
// Returns:   number of clipped values
template <typename T>
int MLMatrix<T>::clipToZero (float threshold)
{
  int nbClip = 0;
  threshold = abs(threshold);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (abs(mat[i][j]) <= threshold) {
        mat[i][j] = 0.0f;
        ++nbClip;
      }
    }
  }
  return nbClip;
}

// Set all values less than threshold to threshold
template <typename T>
int MLMatrix<T>::clipMin (float threshold)
{
  int nbClip = 0;
  threshold = abs(threshold);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (abs(mat[i][j]) < threshold && mat[i][j] >= 0) {
        mat[i][j] = threshold;
        ++nbClip;
      }
      if (abs(mat[i][j]) < threshold && mat[i][j] < 0) {
        mat[i][j] = -threshold;
        ++nbClip;
      }
    }
  }
  return nbClip;
}

// Set all values greater than threshold to threshold
// Leads to:     -threshold < value < threshold
template <typename T>
int MLMatrix<T>::clipMax (float threshold)
{
  int nbClip = 0;
  threshold = abs(threshold);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      if (mat[i][j] >  threshold) {
        mat[i][j] =  threshold;
        ++nbClip;
      }
      if (mat[i][j] < -threshold) {
        mat[i][j] = -threshold;
        ++nbClip;
      }
    }
  }
  return nbClip;
}

// Create a matrix with the sign (+1 or -1) of each element of an input matrix
template<typename T>
MLMatrix<T> MLMatrix<T>::sgn()
{
  MLMatrix<T> S(rows, cols, 0);
  for (unsigned i=0; i<rows; ++i) {
    for (unsigned j=0; j<cols; ++j) {
      S(i,j) = (0 > mat[i][j]) ? T(-1) : T(1);
    }
  }
  return S;
}

// Set an entire row or column to zero
template <typename T>
void MLMatrix<T>::setZeroRow(const int rowNumber)
{
  for (int j = 0; j < cols; ++j) this->mat[rowNumber][j] = T(0);
}

template <typename T>
void MLMatrix<T>::setZeroCol(const int colNumber)
{
  for (int i = 0; i < rows; ++i) this->mat[i][colNumber] = T(0);
}

// Set an entire row or column to a given value
template <typename T>
void MLMatrix<T>::setRow(const int rowNumber, const T value)
{
  for (int j = 0; j < cols; ++j) this->mat[rowNumber][j] = value;
}

template <typename T>
void MLMatrix<T>::setCol(const int colNumber, const T value)
{
  for (int i = 0; i < rows; ++i) this->mat[i][colNumber] = value;
}

// Replace an entire row or column with values from a 1D matrix
template <typename T>
void MLMatrix<T>::setRowMat(const int rowNumber, const MLMatrix<T> values)
{
  for (int j = 0; j < cols; ++j) this->mat[rowNumber][j] = values(0, j);
}

template <typename T>
void MLMatrix<T>::setColMat(const int colNumber, MLMatrix<T> values)
{
  for (int i = 0; i < rows; ++i) this->mat[i][colNumber] = values(i, 0);
}

// Apply dropout to a matrix: elements are set to zero with a given probability
// Returns the dropout mask: a matrix of the same size with
// 1 if the element didn't change and 0 if was set to 0
template <typename T>
MLMatrix<uint8_t>  MLMatrix<T>::dropout(const float proba)
{
  MLMatrix<uint8_t> mask(rows, cols, 1);
  for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
      if (float(random(10000)) / 10000.0f < proba) {
        mat[i][j] = T(0);
        mask(i,j) = 0;
     }
  return mask;
}

///////////////////////////////////////////////////////////////
//      Pruning functions
///////////////////////////////////////////////////////////////

// Put the values of the matrix in a vector and sort the vector in descending order
// usage: std::vector<float> vec = M.sortValues();
// argument true (default) if sort on absolute values, false if not
template<typename T>
std::vector<T> MLMatrix<T>::sortValues(bool absVal)
{
  std::vector<T> vec;
  for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j) {
      if (absVal) vec.push_back(abs(mat[i][j]));
      else        vec.push_back(mat[i][j]);     
    }
  sort(vec.begin(), vec.end(), std::greater<T>());
  return vec;
}

// Verify if a row or column is full of 0
template<typename T>
bool MLMatrix<T>::zeroRow(int rowNumber)
{
  bool isZero = false;
  MLMatrix<T> R(1, cols, T(0));
  for (unsigned j=0; j<cols; ++j) R(0,j) = mat[rowNumber][j];
  T valMax = R.max();
  T valMin = R.min();
  isZero = (valMax == T(0)) && (valMin == T(0)) ? true : false;
  return isZero;
}

template<typename T>
bool MLMatrix<T>::zeroCol(int colNumber)
{
  bool isZero = false;
  MLMatrix<T> C(1, rows, T(0));
  for (unsigned i=0; i<rows; ++i) C(0,i) = mat[i][colNumber];
  T valMax = C.max();
  T valMin = C.min();
  isZero = (valMax == T(0)) && (valMin == T(0)) ? true : false;
  return isZero;
}

template<typename T>
uint16_t MLMatrix<T>::countZeroRow(int rowNumber)
{
  uint16_t zero = 0;
  for (unsigned j=0; j<cols; ++j) if(mat[rowNumber][j] == T(0)) ++ zero;
  return zero;
}

template<typename T>
uint16_t MLMatrix<T>::countZeroCol(int colNumber)
{
  uint16_t zero = 0;
  for (unsigned i=0; i<rows; ++i) if(mat[i][colNumber] == T(0)) ++ zero;
  return zero;
}

// Extract a row or a column from a matrix
template<typename T>
MLMatrix<T> MLMatrix<T>::row(const uint16_t rowNumber)
{
  MLMatrix<T> result(1, cols, 0);
  if (rowNumber > rows) { 
    Serial.printf("Row extraction error: row %d greater than %d\n", rowNumber, rows);
    while(1);
  }
  for (unsigned j=0; j<cols; ++j) result(0, j) = mat[rowNumber][j];
  return result;
}

template<typename T>
MLMatrix<T> MLMatrix<T>::col(const uint16_t colNumber)
{
  MLMatrix<T> result(rows, 1, 0);
  if (colNumber > cols) { 
    Serial.printf("Column extraction error: col %d greater than %d\n", colNumber, cols);
    while(1);
  }
  for (unsigned i=0; i<rows; ++i) result(i, 0) = mat[i][colNumber];
  return result;
}

/* Extract a submatrix
      row0 <= row < row0 + nrows
      col0 <= col < col0 + ncols
*/
template<typename T>
MLMatrix<T> MLMatrix<T>::subMatrix(const uint16_t row0, const uint16_t nrows, const uint16_t col0, const uint16_t ncols)
{
  if (row0 + nrows > rows || col0 + ncols > cols) {
    Serial.printf ("Submatrix extraction error (rows from %d to %d, cols from %d to %d)", row0, row0+nrows, col0, col0+ncols);
    while(1);
  }
  MLMatrix<T> result(nrows, ncols, 0);
  for (unsigned i=0; i<nrows; ++i) 
    for (unsigned j=0; j<ncols; ++j) 
      result(i, j) = mat[row0 + i][col0 + j];
  return result;
}

// Remove a row from the matrix
// usage A.removeRow(n);
template<typename T>
MLMatrix<T> MLMatrix<T>::removeRow(const uint16_t index)
{
  if (index > rows) {
    Serial.printf ("Remove error: cannot remove row %d, size is %d\n", index, rows);
    while(1);
  }
  for (unsigned i=index; i<rows-1; ++i)
    for (unsigned j=0; j<cols; ++j)
      mat[i][j] = mat[i+1][j];
  --rows;
  return *this;
}

// Remove a column from the matrix
// usage A.removeCol(n);
template<typename T>
MLMatrix<T> MLMatrix<T>::removeCol(const uint16_t index)
{
  if (index > cols) {
    Serial.printf ("Remove error: cannot remove col %d, size is %d\n", index, cols);
    while(1);
  }
  for (unsigned i=0; i<rows; ++i)
    for (unsigned j=index; j<cols-1; ++j)
      mat[i][j] = mat[i][j+1];
  --cols;
  return *this;
}

///////////////////////////////////////////////////////////////
/*
  Specific methods for the DeepShift algorithm
  https://arxiv.org/abs/1905.13298#
*/
///////////////////////////////////////////////////////////////

// Returns the integral values nearest to the elements of a matrix, with halfway cases rounded away from zero.
// Adds 'shift' (default is 10) to the elements of the matrix
template<typename T>
MLMatrix<uint8_t> MLMatrix<T>::matRound(uint8_t shift)
{
  MLMatrix<uint8_t> result(rows, cols, 0);
  for (unsigned i = 0; i < rows; ++i) {
    for (unsigned j = 0; j < cols; ++j) {
      if (mat[i][j] < T(-shift)) mat[i][j] = T(-shift);
      if (mat[i][j] > T( shift)) mat[i][j] = T( shift);
      result(i,j) = round(this->mat[i][j] + shift);
    }
  }
  return result;
}
#endif
