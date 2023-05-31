// Auto-generated. Do not edit!

// (in-package scara_command.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class ScaraHomoMatrixRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.q1 = null;
      this.q2 = null;
      this.q3 = null;
    }
    else {
      if (initObj.hasOwnProperty('q1')) {
        this.q1 = initObj.q1
      }
      else {
        this.q1 = 0.0;
      }
      if (initObj.hasOwnProperty('q2')) {
        this.q2 = initObj.q2
      }
      else {
        this.q2 = 0.0;
      }
      if (initObj.hasOwnProperty('q3')) {
        this.q3 = initObj.q3
      }
      else {
        this.q3 = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ScaraHomoMatrixRequest
    // Serialize message field [q1]
    bufferOffset = _serializer.float32(obj.q1, buffer, bufferOffset);
    // Serialize message field [q2]
    bufferOffset = _serializer.float32(obj.q2, buffer, bufferOffset);
    // Serialize message field [q3]
    bufferOffset = _serializer.float32(obj.q3, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ScaraHomoMatrixRequest
    let len;
    let data = new ScaraHomoMatrixRequest(null);
    // Deserialize message field [q1]
    data.q1 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [q2]
    data.q2 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [q3]
    data.q3 = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/ScaraHomoMatrixRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '79ad2c92f4e03a043c435fd450b7abbe';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 q1
    float32 q2
    float32 q3
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ScaraHomoMatrixRequest(null);
    if (msg.q1 !== undefined) {
      resolved.q1 = msg.q1;
    }
    else {
      resolved.q1 = 0.0
    }

    if (msg.q2 !== undefined) {
      resolved.q2 = msg.q2;
    }
    else {
      resolved.q2 = 0.0
    }

    if (msg.q3 !== undefined) {
      resolved.q3 = msg.q3;
    }
    else {
      resolved.q3 = 0.0
    }

    return resolved;
    }
};

class ScaraHomoMatrixResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.A1 = null;
      this.A2 = null;
      this.A3 = null;
    }
    else {
      if (initObj.hasOwnProperty('A1')) {
        this.A1 = initObj.A1
      }
      else {
        this.A1 = new std_msgs.msg.Float32MultiArray();
      }
      if (initObj.hasOwnProperty('A2')) {
        this.A2 = initObj.A2
      }
      else {
        this.A2 = new std_msgs.msg.Float32MultiArray();
      }
      if (initObj.hasOwnProperty('A3')) {
        this.A3 = initObj.A3
      }
      else {
        this.A3 = new std_msgs.msg.Float32MultiArray();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ScaraHomoMatrixResponse
    // Serialize message field [A1]
    bufferOffset = std_msgs.msg.Float32MultiArray.serialize(obj.A1, buffer, bufferOffset);
    // Serialize message field [A2]
    bufferOffset = std_msgs.msg.Float32MultiArray.serialize(obj.A2, buffer, bufferOffset);
    // Serialize message field [A3]
    bufferOffset = std_msgs.msg.Float32MultiArray.serialize(obj.A3, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ScaraHomoMatrixResponse
    let len;
    let data = new ScaraHomoMatrixResponse(null);
    // Deserialize message field [A1]
    data.A1 = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset);
    // Deserialize message field [A2]
    data.A2 = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset);
    // Deserialize message field [A3]
    data.A3 = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Float32MultiArray.getMessageSize(object.A1);
    length += std_msgs.msg.Float32MultiArray.getMessageSize(object.A2);
    length += std_msgs.msg.Float32MultiArray.getMessageSize(object.A3);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/ScaraHomoMatrixResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7d3246727dfc9f450b1b9de106536d12';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Float32MultiArray A1
    std_msgs/Float32MultiArray A2
    std_msgs/Float32MultiArray A3
    
    
    ================================================================================
    MSG: std_msgs/Float32MultiArray
    # Please look at the MultiArrayLayout message definition for
    # documentation on all multiarrays.
    
    MultiArrayLayout  layout        # specification of data layout
    float32[]         data          # array of data
    
    
    ================================================================================
    MSG: std_msgs/MultiArrayLayout
    # The multiarray declares a generic multi-dimensional array of a
    # particular data type.  Dimensions are ordered from outer most
    # to inner most.
    
    MultiArrayDimension[] dim # Array of dimension properties
    uint32 data_offset        # padding elements at front of data
    
    # Accessors should ALWAYS be written in terms of dimension stride
    # and specified outer-most dimension first.
    # 
    # multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
    #
    # A standard, 3-channel 640x480 image with interleaved color channels
    # would be specified as:
    #
    # dim[0].label  = "height"
    # dim[0].size   = 480
    # dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
    # dim[1].label  = "width"
    # dim[1].size   = 640
    # dim[1].stride = 3*640 = 1920
    # dim[2].label  = "channel"
    # dim[2].size   = 3
    # dim[2].stride = 3
    #
    # multiarray(i,j,k) refers to the ith row, jth column, and kth channel.
    
    ================================================================================
    MSG: std_msgs/MultiArrayDimension
    string label   # label of given dimension
    uint32 size    # size of given dimension (in type units)
    uint32 stride  # stride of given dimension
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ScaraHomoMatrixResponse(null);
    if (msg.A1 !== undefined) {
      resolved.A1 = std_msgs.msg.Float32MultiArray.Resolve(msg.A1)
    }
    else {
      resolved.A1 = new std_msgs.msg.Float32MultiArray()
    }

    if (msg.A2 !== undefined) {
      resolved.A2 = std_msgs.msg.Float32MultiArray.Resolve(msg.A2)
    }
    else {
      resolved.A2 = new std_msgs.msg.Float32MultiArray()
    }

    if (msg.A3 !== undefined) {
      resolved.A3 = std_msgs.msg.Float32MultiArray.Resolve(msg.A3)
    }
    else {
      resolved.A3 = new std_msgs.msg.Float32MultiArray()
    }

    return resolved;
    }
};

module.exports = {
  Request: ScaraHomoMatrixRequest,
  Response: ScaraHomoMatrixResponse,
  md5sum() { return 'b04a28e6794c58d88ef84b160d32201a'; },
  datatype() { return 'scara_command/ScaraHomoMatrix'; }
};
