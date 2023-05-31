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


//-----------------------------------------------------------

class ScaraVelIKRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.q1 = null;
      this.q2 = null;
      this.q3 = null;
      this.Vx = null;
      this.Vy = null;
      this.Vz = null;
      this.Wx = null;
      this.Wy = null;
      this.Wz = null;
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
      if (initObj.hasOwnProperty('Vx')) {
        this.Vx = initObj.Vx
      }
      else {
        this.Vx = 0.0;
      }
      if (initObj.hasOwnProperty('Vy')) {
        this.Vy = initObj.Vy
      }
      else {
        this.Vy = 0.0;
      }
      if (initObj.hasOwnProperty('Vz')) {
        this.Vz = initObj.Vz
      }
      else {
        this.Vz = 0.0;
      }
      if (initObj.hasOwnProperty('Wx')) {
        this.Wx = initObj.Wx
      }
      else {
        this.Wx = 0.0;
      }
      if (initObj.hasOwnProperty('Wy')) {
        this.Wy = initObj.Wy
      }
      else {
        this.Wy = 0.0;
      }
      if (initObj.hasOwnProperty('Wz')) {
        this.Wz = initObj.Wz
      }
      else {
        this.Wz = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ScaraVelIKRequest
    // Serialize message field [q1]
    bufferOffset = _serializer.float32(obj.q1, buffer, bufferOffset);
    // Serialize message field [q2]
    bufferOffset = _serializer.float32(obj.q2, buffer, bufferOffset);
    // Serialize message field [q3]
    bufferOffset = _serializer.float32(obj.q3, buffer, bufferOffset);
    // Serialize message field [Vx]
    bufferOffset = _serializer.float32(obj.Vx, buffer, bufferOffset);
    // Serialize message field [Vy]
    bufferOffset = _serializer.float32(obj.Vy, buffer, bufferOffset);
    // Serialize message field [Vz]
    bufferOffset = _serializer.float32(obj.Vz, buffer, bufferOffset);
    // Serialize message field [Wx]
    bufferOffset = _serializer.float32(obj.Wx, buffer, bufferOffset);
    // Serialize message field [Wy]
    bufferOffset = _serializer.float32(obj.Wy, buffer, bufferOffset);
    // Serialize message field [Wz]
    bufferOffset = _serializer.float32(obj.Wz, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ScaraVelIKRequest
    let len;
    let data = new ScaraVelIKRequest(null);
    // Deserialize message field [q1]
    data.q1 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [q2]
    data.q2 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [q3]
    data.q3 = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Vx]
    data.Vx = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Vy]
    data.Vy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Vz]
    data.Vz = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Wx]
    data.Wx = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Wy]
    data.Wy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Wz]
    data.Wz = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 36;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/ScaraVelIKRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '9a7f08bc2df9b68690007784109f3edc';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 q1
    float32 q2
    float32 q3
    float32 Vx
    float32 Vy
    float32 Vz
    float32 Wx
    float32 Wy
    float32 Wz
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ScaraVelIKRequest(null);
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

    if (msg.Vx !== undefined) {
      resolved.Vx = msg.Vx;
    }
    else {
      resolved.Vx = 0.0
    }

    if (msg.Vy !== undefined) {
      resolved.Vy = msg.Vy;
    }
    else {
      resolved.Vy = 0.0
    }

    if (msg.Vz !== undefined) {
      resolved.Vz = msg.Vz;
    }
    else {
      resolved.Vz = 0.0
    }

    if (msg.Wx !== undefined) {
      resolved.Wx = msg.Wx;
    }
    else {
      resolved.Wx = 0.0
    }

    if (msg.Wy !== undefined) {
      resolved.Wy = msg.Wy;
    }
    else {
      resolved.Wy = 0.0
    }

    if (msg.Wz !== undefined) {
      resolved.Wz = msg.Wz;
    }
    else {
      resolved.Wz = 0.0
    }

    return resolved;
    }
};

class ScaraVelIKResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
      this.q1_dot = null;
      this.q2_dot = null;
      this.q3_dot = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
      if (initObj.hasOwnProperty('q1_dot')) {
        this.q1_dot = initObj.q1_dot
      }
      else {
        this.q1_dot = 0.0;
      }
      if (initObj.hasOwnProperty('q2_dot')) {
        this.q2_dot = initObj.q2_dot
      }
      else {
        this.q2_dot = 0.0;
      }
      if (initObj.hasOwnProperty('q3_dot')) {
        this.q3_dot = initObj.q3_dot
      }
      else {
        this.q3_dot = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ScaraVelIKResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [q1_dot]
    bufferOffset = _serializer.float32(obj.q1_dot, buffer, bufferOffset);
    // Serialize message field [q2_dot]
    bufferOffset = _serializer.float32(obj.q2_dot, buffer, bufferOffset);
    // Serialize message field [q3_dot]
    bufferOffset = _serializer.float32(obj.q3_dot, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ScaraVelIKResponse
    let len;
    let data = new ScaraVelIKResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [q1_dot]
    data.q1_dot = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [q2_dot]
    data.q2_dot = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [q3_dot]
    data.q3_dot = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 13;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/ScaraVelIKResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7a1167ccfb083ccdaf6e453d6a0272a3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    float32 q1_dot
    float32 q2_dot
    float32 q3_dot
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ScaraVelIKResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    if (msg.q1_dot !== undefined) {
      resolved.q1_dot = msg.q1_dot;
    }
    else {
      resolved.q1_dot = 0.0
    }

    if (msg.q2_dot !== undefined) {
      resolved.q2_dot = msg.q2_dot;
    }
    else {
      resolved.q2_dot = 0.0
    }

    if (msg.q3_dot !== undefined) {
      resolved.q3_dot = msg.q3_dot;
    }
    else {
      resolved.q3_dot = 0.0
    }

    return resolved;
    }
};

module.exports = {
  Request: ScaraVelIKRequest,
  Response: ScaraVelIKResponse,
  md5sum() { return '273c8e596b5ffa32d13cf21237d3aaf2'; },
  datatype() { return 'scara_command/ScaraVelIK'; }
};
